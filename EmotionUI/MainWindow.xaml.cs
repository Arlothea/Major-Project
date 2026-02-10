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
using System.Net.Http;

namespace EmotionUI
{
  public partial class MainWindow : Window
  {
    private Task? cameraTask;
    private bool pythonInitialized = false;
    private bool isRunning = false;
    private string _lastShownEmotion = "";
    private int _lastShownPercent = -1;
    private readonly Dictionary<string, int> _bestPercentByEmotion = new Dictionary<string, int>();
    private Process? serverProcess;

    public MainWindow()
    {
      InitializeComponent();
      LoadStudentNamesOnStartup();

      CameraHost.SizeChanged += (s, e) =>
      {
        CameraFeed.Clip = new System.Windows.Media.RectangleGeometry(
            new System.Windows.Rect(0,0, CameraHost.ActualWidth, CameraHost.ActualHeight),
            12, 12
        );
      };
    }
    private async void Start_Click(object sender, RoutedEventArgs e)
    {
      if (isRunning)
      return;

      string pythonHome = @"C:\Program Files\Python39";
      string pythonDll = Path.Combine(pythonHome, "python39.dll");
      string pythonProject = @"C:\Users\Adam Wingell\Documents\Uni Work\Year 3\Dissertation\Major Project\Emotion App";

      try
      {
        if (serverProcess == null || serverProcess.HasExited)
        {
          serverProcess = Process.Start(new ProcessStartInfo
          {
            FileName = @"C:\Program Files\Python39\python.exe",
            Arguments = "\"C:\\Users\\Adam Wingell\\Documents\\Uni Work\\Year 3\\Dissertation\\Major Project\\Emotion App\\server.py\"",
            WorkingDirectory = @"C:\Users\Adam Wingell\Documents\Uni Work\Year 3\Dissertation\Major Project\Emotion App",
            UseShellExecute = false,
            CreateNoWindow = true
          });
          bool serverReady = await WaitForServer();

          if (!serverReady)
          {
            MessageBox.Show("Server failed to start.");
            return;
          }
        }

        if (!pythonInitialized)
        {
            Runtime.PythonDLL = pythonDll;
            PythonEngine.PythonHome = pythonHome;

            string dlls = Path.Combine(pythonHome, "DLLs");
            string site = Path.Combine(pythonHome, "Lib", "site-packages");
            string currentPath = Environment.GetEnvironmentVariable("PATH") ?? "";
            Environment.SetEnvironmentVariable("PATH", $"{pythonHome};{dlls};{site};{currentPath}");

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
                PythonEngine.Initialize();
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
              dynamic cv2 = Py.Import("cv2");
              string cv2ver = cv2.__version__.ToString();
              dynamic mainPy = Py.Import("Main");

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

        isRunning = true;
        StatusText.Text = "Status: Running";
        StatusText.Foreground = System.Windows.Media.Brushes.LightGreen;

        cameraTask = Task.Run(() => RunPythonCamera());
      }
      catch (Exception ex)
      {
        MessageBox.Show("Python failed: " + ex.ToString());
      }
    }

    private void Stop_Click(object sender, RoutedEventArgs e)
    {
      isRunning = false;

      Application.Current.Dispatcher.Invoke(() =>
      {
        CameraFeed.Source = null;     
        EmotionText.Text = "No emotion detected";
        StatusText.Text = "Status: Stopped";
        StatusText.Foreground = System.Windows.Media.Brushes.Red;
      });
    }
    private void Remove_Click(object sender, RoutedEventArgs e)
    {
      if (StudentListBox.SelectedItem is string studentName)
      {
        var result = MessageBox.Show($"Are you sure you want to remove {studentName}?",
                                     "Confirm", MessageBoxButton.YesNo);

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
    private async Task<bool> WaitForServer()
    {
      using var client = new HttpClient();

      for (int i = 0; i < 10; i++)
      {
        try
        {
          var response = await client.GetAsync("http://127.0.0.1:8000/ping");
          if (response.IsSuccessStatusCode)
            return true;
        }
        catch { }

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