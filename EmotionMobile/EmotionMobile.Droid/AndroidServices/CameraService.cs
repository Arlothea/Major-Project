using AndroidX.Camera.Core;
using AndroidX.Camera.Lifecycle;
using AndroidX.Core.Content;
using AndroidX.Lifecycle;
using EmotionMobile.Services;
using Java.Util.Concurrent;

[assembly: Dependency(typeof(EmotionMobile.Droid.AndroidServices.CameraService))]
namespace EmotionMobile.Droid.AndroidServices;

public class CameraService : Java.Lang.Object, ICameraService
{

    bool _cameraStarted;

    public event Action<byte[], int, int> FrameReady;

    ProcessCameraProvider? cameraProvider;
    ImageAnalysis? imageAnalysis;

    void BindCamera()
    {
        imageAnalysis = new ImageAnalysis.Builder()
            .SetBackpressureStrategy(ImageAnalysis.StrategyKeepOnlyLatest)
            .Build();

        imageAnalysis.SetAnalyzer(
            Executors.NewSingleThreadExecutor(),
            new FrameAnalyzer(this)
        );

        var cameraSelector = new CameraSelector.Builder()
            .RequireLensFacing(CameraSelector.LensFacingFront)
            .Build();

        var activity = Platform.CurrentActivity;
        if (activity is not ILifecycleOwner lifecycleOwner)
        {
            return;
        }


        cameraProvider.BindToLifecycle(
            lifecycleOwner,
            cameraSelector,
            imageAnalysis
        );

    }

    public void StartCamera()
    {
        if (_cameraStarted)
            return;

        if (Platform.CurrentActivity == null)
            return;

        _cameraStarted = true;

        var context = Android.App.Application.Context;
        var cameraProviderFuture = ProcessCameraProvider.GetInstance(context);

        cameraProviderFuture.AddListener(
            new Java.Lang.Runnable(() =>
            {
                cameraProvider = (ProcessCameraProvider)cameraProviderFuture.Get();
                BindCamera();
            }),
            ContextCompat.GetMainExecutor(context)
        );
    }


    public void StopCamera()
    {
        cameraProvider?.UnbindAll();
    }

    class FrameAnalyzer : Java.Lang.Object, ImageAnalysis.IAnalyzer
    {
        readonly CameraService service;

        public FrameAnalyzer(CameraService service)
        {
            this.service = service;
        }

        public void Analyze(IImageProxy image)
        {
            var buffer = image.GetPlanes()[0].Buffer;
            byte[] data = new byte[buffer.Remaining()];
            buffer.Get(data);

            service.FrameReady?.Invoke(data, image.Width, image.Height);
            image.Close();
        }
    }
}
