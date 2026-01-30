using Plugin.LocalNotification;
using System.Text.Json;

namespace EmotionMobile;

public partial class MainPage : ContentPage
{

    private const string API_URL = "http://192.168.0.125:8000/current";

    private string lastAlertTime = "";

    public MainPage()
    {
        InitializeComponent();

        _ = LocalNotificationCenter.Current.RequestNotificationPermission();

        Dispatcher.StartTimer(TimeSpan.FromSeconds(1), () =>
        {
            _ = CheckForAlerts();
            return true;
        });
    }

    protected override async void OnAppearing()
    {
        base.OnAppearing();
        await RefreshUIFromServer();
    }

    private async Task CheckForAlerts()
    {
        try
        {
            var json = await new HttpClient().GetStringAsync(API_URL);

            if (string.IsNullOrWhiteSpace(json) || json == "null")
                return;

            var alert = JsonSerializer.Deserialize<Alert>(json);
            if (alert == null) return;
            
            UpdateUI(alert);

            if (alert.time != lastAlertTime)
            {
                lastAlertTime = alert.time;
                await ShowEscalation(alert);
            }
        }
        catch { }
    }

    private async Task RefreshUIFromServer()
    {
        try
        {
            var json = await new HttpClient().GetStringAsync(API_URL);
            if (string.IsNullOrWhiteSpace(json) || json == "null") return;

            var alert = JsonSerializer.Deserialize<Alert>(json);
            if (alert != null) UpdateUI(alert);
        }
        catch { }
    }

    private void UpdateUI(Alert alert)
    {
        MainThread.BeginInvokeOnMainThread(() =>
        {
            NameLabel.Text = alert.name;
            CameraLabel.Text = $"Location: {alert.camera}";
            LevelLabel.Text = $"Escalation level: {alert.level}";
        });
    }

    private async Task ShowEscalation(Alert alert)
    {
        var request = new NotificationRequest
        {
            NotificationId = 100,
            Title = "🚨 Escalation Detected",
            Description = $"{alert.name} – {alert.camera} (Level {alert.level})",
            Schedule = new NotificationRequestSchedule
            {
                NotifyTime = DateTime.Now
            }
        };

        await LocalNotificationCenter.Current.Show(request);
    }
}


