using Microsoft.Win32;
using Python.Runtime;
using System;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;

namespace EmotionUI
{
  public partial class MainWindow : Window
  {
    // Background task for running the Python camera loop
    private Task? cameraTask;
    // Tracking if Python has been initialized to avoid multiple initializations
    private bool pythonInitialized = false;
    // Main on/off switch for the camera loop
    private bool isRunning = false;
    // used to reduce UI updates - only update if emotion or confidence changes
    private string _lastShownEmotion = "";
    private int _lastShownPercent = -1;
    // tracks the best (highest) confidence percentage seen for each emotion to avoid UI regression
    private readonly Dictionary<string, int> _bestPercentByEmotion = new Dictionary<string, int>();
    // Process reference for the Python server to allow proper cleanup on exit
    private Process? serverProcess;

    public MainWindow()
    {
      InitializeComponent();
      // Load existing student names from the known_faces directory on startup
      LoadStudentNamesOnStartup();
      // Whenever the host camera is resized update the clipping region of the camera feed to maintain rounded corners
      CameraHost.SizeChanged += (s, e) =>
      {
        CameraFeed.Clip = new System.Windows.Media.RectangleGeometry(
            new System.Windows.Rect(0,0, CameraHost.ActualWidth, CameraHost.ActualHeight),
            12, 12
        );
      };
    }
    /* -------------------------- START BUTTON CLICK HANDLER --------------------------
    * 1. Starts the Python server if not already running
    * 2. Initializes Python.NET if not already initialized
    * 3. Imports necessary Python modules to ensure they're available before starting the camera loop
    * 4. Starts the background task that runs the camera loop which continuously fetches frames and emotion data from Python and updates the UI
    */
    private async void Start_Click(object sender, RoutedEventArgs e)
    {
      // If camera is already running, do nothing
      if (isRunning)
      return;
      // Define paths for Python home, DLL, and project
      string pythonHome = @"C:\Program Files\Python39";
      string pythonDll = Path.Combine(pythonHome, "python39.dll");
      string pythonProject = @"C:\Users\Adam Wingell\Documents\Uni Work\Year 3\Dissertation\Major Project\Emotion App";

      try
      {
        /*-------------------------- SERVER STARTUP --------------------------*/
        if (serverProcess == null || serverProcess.HasExited)
        {
          serverProcess = Process.Start(new ProcessStartInfo
          {
            FileName = @"C:\Program Files\Python39\python.exe",
            Arguments = "\"C:\\Users\\Adam Wingell\\Documents\\Uni Work\\Year 3\\Dissertation\\Major Project\\Emotion App\\server.py\"",
            WorkingDirectory = @"C:\Users\Adam Wingell\Documents\Uni Work\Year 3\Dissertation\Major Project\Emotion App",
            // use shell execute false to allow redirection and better control
            UseShellExecute = false,
            // Hide the console window
            CreateNoWindow = true
          });
          // wait for the server to start and respond to a ping before proceeding to initialize Python.NET and the camera loop
          bool serverReady = await WaitForServer();

          if (!serverReady)
          {
            MessageBox.Show("Server failed to start.");
            return;
          }
        }
        /*-------------------------- PYTHON.NET INITIALIZATION --------------------------*/
        if (!pythonInitialized)
        {
          Runtime.PythonDLL = pythonDll;
          PythonEngine.PythonHome = pythonHome;

          // This allows Python to find the necessary DLLs and packages by adding them to the PATH environment variable at runtime
          string dlls = Path.Combine(pythonHome, "DLLs");
          string site = Path.Combine(pythonHome, "Lib", "site-packages");
          // Ensures that the Python project directory is also in the PATH so that imports from the project work correctly
          string currentPath = Environment.GetEnvironmentVariable("PATH") ?? "";
          Environment.SetEnvironmentVariable("PATH", $"{pythonHome};{dlls};{site};{currentPath}");

          // Set the PythonPath to include the project and standard library paths so that imports in Python.NET work correctly
          PythonEngine.PythonPath = string.Join(
            Path.PathSeparator.ToString(),
            new string[]
            {
                pythonProject,
                Path.Combine(pythonProject, "Runtime"),
                Path.Combine(pythonProject, "Models"),
                Path.Combine(pythonHome, "Lib"),
                Path.Combine(pythonHome, "Lib", "site-packages"),
                Path.Combine(pythonHome, "DLLs")
            });

          try
          {
            // Initialize the Python engine
            PythonEngine.Initialize();
            // Allow threads to use Python(necessary for the background camera task).
            PythonEngine.BeginAllowThreads();

            pythonInitialized = true;
          }
          catch (Exception initEx)
          {
            MessageBox.Show("PythonEngine initialization failed:\n" + initEx.ToString());
            return;
          }
        }

        try
        {
          using (Py.GIL())
          {
            dynamic sys = Py.Import("sys");
            try
            {
              dynamic cv2 = Py.Import("cv2"); // verifies OpenCV can be imported correctly before starting the camera loop
              string cv2ver = cv2.__version__.ToString();
              dynamic mainPy = Py.Import("Main"); // verifies that the Main module can be imported and is working before starting the camera loop

            }
            catch (Exception cvEx)
            {
              MessageBox.Show("Import cv2 failed:\n" + cvEx.ToString());
              return;
            }
          }
        }
        catch (Exception exGIL)
        {
          MessageBox.Show("GIL/import test failed:\n" + exGIL.ToString());
          return;
        }
        /* -------------------------- UI STATE  --------------------------*/
        // Starts the camera loop in a background task so that it doesn't block the UI thread.
        isRunning = true;
        StatusText.Text = "Status: Running";
        StatusText.Foreground = System.Windows.Media.Brushes.LightGreen;

        // RunPythonCamera() loops while isRunning is true, continuously fetching frames and emotion data from the Python backend and updating the UI.
        cameraTask = Task.Run(() => RunPythonCamera());
      }
      catch (Exception ex)
      {
        MessageBox.Show("Python failed: " + ex.ToString());
      }
    }
    /* -------------------------- STOP BUTTON CLICK HANDLER -------------------------- */
    // flips isRunning to false which signals the camera loop to stop and then resets the UI to a default state
    // indicating that the camera is stopped and no emotion is detected.
    private void Stop_Click(object sender, RoutedEventArgs e)
    {
      isRunning = false;
      // Wait for the camera task to finish to ensure we don't have background threads still running that are trying to access the UI after we've reset it.
      Application.Current.Dispatcher.Invoke(() =>
      {
        CameraFeed.Source = null;     
        EmotionText.Text = "No emotion detected";
        StatusText.Text = "Status: Stopped";
        StatusText.Foreground = System.Windows.Media.Brushes.Red;
      });
    }
    /* -------------------------- REMOVE SELECTED CHILD FROM KNOWN_FACES FOLDER --------------------------*/
    private void Remove_Click(object sender, RoutedEventArgs e)
    {
      if (StudentListBox.SelectedItem is string studentName)
      {
        var result = MessageBox.Show($"Are you sure you want to remove {studentName}?",
                                     "Confirm", MessageBoxButton.YesNo);
        // If the user confirms, delete the corresponding image file from the known_faces folder and remove the name from the list box.
        if (result == MessageBoxResult.Yes)
        {
          string knownFacesFolder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "known_faces");
          string fileToDelete = Directory.GetFiles(knownFacesFolder, "*.jpg")
                                         .FirstOrDefault(f => Path.GetFileNameWithoutExtension(f)
                                         .Equals(studentName, StringComparison.OrdinalIgnoreCase));

          if (fileToDelete != null)
          {
            File.Delete(fileToDelete);
            StudentListBox.Items.Remove(studentName);
            // If there are no more students in the list, show the "No faces" text.
            if (StudentListBox.Items.Count == 0)
              NoFacesText.Visibility = Visibility.Visible;
          }
        }
      }
      else
      {
        MessageBox.Show("Please select a student to remove.");
      }
    }
    /* -------------------------- HELPER METHOD TO WAIT FOR THE PYTHON SERVER TO START --------------------------*/
    private async Task<bool> WaitForServer()
    {
      using var client = new HttpClient();
      // Try to ping the server up to 10 times. If we get a successful response, the server is ready.
      for (int i = 0; i < 10; i++)
      {
        try
        {
          var response = await client.GetAsync("http://127.0.0.1:8000/ping");
          if (response.IsSuccessStatusCode)
            return true;
        }
        catch 
        {
          // Server not ready yet, wait and try again
        }

        await Task.Delay(500);
      }

      return false;
    }
  
  private void RunPythonCamera()
  {
    dynamic cv2 = null;
    dynamic mainPy = null;

    try
    {
      using (Py.GIL())
      {
        cv2 = Py.Import("cv2");
        mainPy = Py.Import("Main");
      }

      while (isRunning)
      {
        byte[] bytes = null;
        string uiText = null;

        using (Py.GIL())
        {
          var result = mainPy.get_frame();

          if (result == null || result.__len__() < 2)
          {
          continue;
          }

          dynamic frame = result[0];
          dynamic emotions = result[1];


          if (frame != null)
          {
            dynamic encoded = cv2.imencode(".jpg", frame, new PyList(new PyObject[]
            {
              new PyInt((int)cv2.IMWRITE_JPEG_QUALITY),
              new PyInt(75)
            }))[1];

            bytes = ((PyObject)encoded.tobytes()).As<byte[]>();

            int count = (int)emotions.__len__();
            if (count > 0)
            {
              string emotion = emotions[0][0].ToString();

              double confidence = double.Parse(emotions[0][1].ToString(), System.Globalization.CultureInfo.InvariantCulture);
              int percent = (int)Math.Round(confidence * 100);

              if (_bestPercentByEmotion.TryGetValue(emotion, out int bestSoFar))
              {
                if (percent > bestSoFar)
                  _bestPercentByEmotion[emotion] = percent;
                else
                  percent = bestSoFar;
              }
              else
              {
                _bestPercentByEmotion[emotion] = percent;
              }

              bool shouldUpdate = (emotion != _lastShownEmotion) || (percent != _lastShownPercent);

              if (shouldUpdate)
              {
                _lastShownEmotion = emotion;
                _lastShownPercent = percent;

                uiText = $"Detected Emotion: {emotion} ({percent}%)";
              }
            }
          }
          if (bytes != null && bytes.Length > 0)
          {
            Application.Current.Dispatcher.BeginInvoke(() =>
            {
              using (var ms = new MemoryStream(bytes))
              {
                var bitmap = new BitmapImage();
                bitmap.BeginInit();
                bitmap.StreamSource = ms;
                bitmap.CacheOption = BitmapCacheOption.OnLoad;
                bitmap.EndInit();
                bitmap.Freeze();
                CameraFeed.Source = bitmap;
              }
              if (uiText != null && EmotionText.Text != uiText)
                EmotionText.Text = uiText;
            });
          }
          Thread.Sleep(15);
        }
      }
    }
    finally
    {
      try
      {
        using (Py.GIL())
        {
          mainPy?.release();
        }
      }
      catch { }
    }
  }
    private void Upload_Click(object sender, RoutedEventArgs e)
    {
      OpenFileDialog dialog = new OpenFileDialog
      {
        Title = "Select Face Image",
        Filter = "Image Files|*.jpg;*.jpeg;*.png"
      };

      if (dialog.ShowDialog() == true)
      {
        string sourcePath = dialog.FileName;
        string fileName = Path.GetFileName(sourcePath);

        string knownFacesFolder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "known_faces");
        Directory.CreateDirectory(knownFacesFolder);
        string destinationPath = Path.Combine(knownFacesFolder, fileName);

        File.Copy(sourcePath, destinationPath, true);

        string studentName = Path.GetFileNameWithoutExtension(fileName);
        StudentListBox.Items.Add(studentName);
        NoFacesText.Visibility = Visibility.Collapsed;
      }
    }
    private void LoadStudentNamesOnStartup()
    {
      string knownFacesFolder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "known_faces");
      if (Directory.Exists(knownFacesFolder))
      {
        var files = Directory.GetFiles(knownFacesFolder, "*.jpg");
        foreach (var file in files)
        {
          StudentListBox.Items.Add(Path.GetFileNameWithoutExtension(file));
        }

        NoFacesText.Visibility = StudentListBox.Items.Count == 0 ? Visibility.Visible : Visibility.Collapsed;
      }
    }

    protected override void OnClosed(EventArgs e)
    {
      isRunning = false;

        try
        {
            cameraTask.Wait();
        }
        catch { }

        if (PythonEngine.IsInitialized)
        {
            using (Py.GIL()) { }
            PythonEngine.Shutdown();
        }
        base.OnClosed(e);

        if (serverProcess != null && !serverProcess.HasExited)
        {
          serverProcess.Kill();
        }

    }
  }
}