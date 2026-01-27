using Plugin.LocalNotification;

namespace EmotionMobile;

public partial class MainPage : ContentPage
{
    public MainPage()
    {
        InitializeComponent();
    }

    private async void OnSendAlert(object sender, EventArgs e)
    {
        var request = new NotificationRequest
        {
            NotificationId = 100,
            Title = "Emotion Alert",
            Description = "Bubbling detected – intervention recommended.",
            Schedule = new NotificationRequestSchedule
            {
                NotifyTime = DateTime.Now.AddSeconds(1)
            }
        };

        await LocalNotificationCenter.Current.Show(request);
    }
}

