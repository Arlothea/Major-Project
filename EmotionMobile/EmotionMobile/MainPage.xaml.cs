using EmotionMobile.Services;

namespace EmotionMobile;

public partial class MainPage : ContentPage
{
    ICameraService cameraService;
    bool _cameraStarted;

    public MainPage()
    {
        InitializeComponent();

        cameraService = DependencyService.Get<ICameraService>();

        if (cameraService != null)
        {
            cameraService.FrameReady += OnFrameReady;
        }
    }
    protected override void OnAppearing()
    {
        base.OnAppearing();

#if ANDROID
        if (_cameraStarted)
            return;

        if (cameraService == null)
            return;

        if (AndroidX.Core.Content.ContextCompat.CheckSelfPermission(
                Android.App.Application.Context,
                Android.Manifest.Permission.Camera)
            == Android.Content.PM.Permission.Granted)
        {
            _cameraStarted = true;
            cameraService.StartCamera();
        }
#endif
    }
    protected override void OnDisappearing()
    {
        if (_cameraStarted && cameraService != null)
        {
            cameraService.StopCamera();
            _cameraStarted = false;
        }

        base.OnDisappearing();
    }
    void OnFrameReady(byte[] frame, int width, int height)
    {
    }
}

