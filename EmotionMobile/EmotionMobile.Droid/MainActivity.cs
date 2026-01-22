using Android;
using Android.App;
using Android.Content.PM;
using Android.OS;
using EmotionMobile;
using Microsoft.Maui.Controls;

[Activity(
    Theme = "@style/Maui.SplashTheme",
    MainLauncher = true,
    ConfigurationChanges =
        ConfigChanges.ScreenSize |
        ConfigChanges.Orientation |
        ConfigChanges.UiMode |
        ConfigChanges.ScreenLayout |
        ConfigChanges.SmallestScreenSize)]
public class MainActivity : MauiAppCompatActivity
{
    const int CameraPermissionRequestCode = 1001;

    protected override void OnCreate(Bundle? savedInstanceState)
    {
        base.OnCreate(savedInstanceState);

        if (CheckSelfPermission(Manifest.Permission.Camera) != Permission.Granted)
        {
            RequestPermissions(
                new[] { Manifest.Permission.Camera },
                CameraPermissionRequestCode
            );
        }
    }
    public override void OnRequestPermissionsResult(
        int requestCode,
        string[] permissions,
        Permission[] grantResults)
    {
        base.OnRequestPermissionsResult(requestCode, permissions, grantResults);
    }
}

