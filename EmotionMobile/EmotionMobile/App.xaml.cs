using Plugin.LocalNotification;
using Microsoft.Maui.ApplicationModel;

namespace EmotionMobile
{
    public partial class App : Application
    {
        public App()
        {
            InitializeComponent();

            LocalNotificationCenter.Current.NotificationActionTapped += e =>
            {
                if (e.Request.ReturningData == "show_current")
                {
                    MainThread.BeginInvokeOnMainThread(async () =>
                    {
                        await Task.Delay(500);

                        if (Shell.Current != null)
                        {
                            await Shell.Current.GoToAsync("//MainPage");
                        }
                    });
                }
            };
        }

        protected override Window CreateWindow(IActivationState? activationState)
        {
            return new Window(new AppShell());
        }
    }
}
