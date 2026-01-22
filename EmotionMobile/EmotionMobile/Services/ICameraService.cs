namespace EmotionMobile.Services;

public interface ICameraService
{
    void StartCamera();
    void StopCamera();

    event Action<byte[], int, int> FrameReady;
}

