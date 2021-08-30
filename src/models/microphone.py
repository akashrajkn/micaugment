import torch
import torch.nn    as nn
import torch.optim as optim


def smoothmax(a, b):

    exp_a       = torch.exp(a)
    exp_b       = torch.exp(b)
    numerator   = a * exp_a + b * exp_b
    denominator = exp_a + exp_b

    return numerator / denominator


def smoothmin(a, b):
    exp_m_a     = torch.exp(-a)
    exp_m_b     = torch.exp(-b)
    numerator   = a * exp_m_a + b * exp_m_b
    denominator = exp_m_a + exp_m_b

    return numerator / denominator


class MicrophoneModel(nn.Module):

    def __init__(self, hparams):
        super(MicrophoneModel, self).__init__()

        self.win_length = hparams['win_length']
        self.hop_length = hparams['hop_length']
        self.n_fft      = hparams['n_fft']
        self.lr         = hparams['lr']
        self.beta1      = hparams['beta1']
        self.beta2      = hparams['beta2']

        stft_dim        = int(self.n_fft / 2) + 1

        self.impulse_response = nn.Parameter(torch.randn(stft_dim, 1, requires_grad=True))
        self.threshold        = nn.Parameter(torch.randn(stft_dim, 1, requires_grad=True))
        self.filter           = nn.Parameter(torch.randn(stft_dim, 1, requires_grad=True))
        self.mic_clip         = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):

        x_cmplx = torch.stft(x, hop_length=self.hop_length, win_length=self.win_length, n_fft=self.n_fft, return_complex=True)
        y_1     = self.impulse_response * x_cmplx
        y_2     = torch.istft(y_1 * torch.sigmoid(torch.abs(y_1) ** 2 - self.threshold.expand(y_1.size(-2), y_1.size(-1))), n_fft=self.n_fft)
        y_3     = y_2 + torch.istft(torch.stft(torch.randn_like(y_2), hop_length=self.hop_length, win_length=self.win_length, n_fft=self.n_fft, return_complex=True) * self.filter, n_fft=self.n_fft)
        y       = smoothmin(smoothmax(y_3, -self.mic_clip), self.mic_clip)

        return y

    def get_optimizer(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        return optimizer


if __name__ == '__main__':

    hparams = {
        'win_length' : 1024,
        'n_fft'      : 1024,
        'hop_length' : 256,
        'lr'         : 1e-4,
        'beta1'      : 0.5,
        'beta2'      : 0.9,
    }

    audio            = torch.randn(3, 16384)
    microphone_model = MicrophoneModel(hparams)
    audio_mic        = microphone_model(audio)

    print(f'audio out size: {audio_mic.size()}')
    print('DONE')
