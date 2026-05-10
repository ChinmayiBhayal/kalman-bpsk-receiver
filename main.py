import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

try:
    import sounddevice as sd
    HAVE_SOUNDDEVICE = True
except Exception:
    HAVE_SOUNDDEVICE = False


# =========================================================
# CONFIG
# =========================================================
INPUT_WAV = "input.wav"
OUTPUT_WAV = "recovered_audio.wav"
PLOTS_DIR = "plots"

EBN0_DB = 10
PHASE_OFFSET = np.pi / 4
FREQ_OFFSET = 0.0001
PILOT_LEN = 100

Q_PHASE = 1e-5
Q_FREQ = 1e-7
R_SCALE = 5.0

os.makedirs(PLOTS_DIR, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================
def normalize_audio(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    peak = np.max(np.abs(x))
    if peak == 0:
        return x
    return x / peak


def play_audio(x: np.ndarray, fs: int, label: str):
    print(label)
    if HAVE_SOUNDDEVICE:
        sd.play(x, fs)
        sd.wait()
    else:
        print("sounddevice not available, skipping playback.")


def audio_to_bits(audio: np.ndarray):
    """
    Convert normalized audio in [-1, 1] to bits via int16 -> uint8 -> bits.
    Returns:
        bits: 0/1 uint8 array
        audio_i16: original int16 samples
    """
    audio_i16 = np.clip(np.round(audio * 32767.0), -32768, 32767).astype("<i2")
    byte_view = audio_i16.view(np.uint8)
    bits = np.unpackbits(byte_view, bitorder="big").astype(np.uint8)
    return bits, audio_i16


def bits_to_audio(bits: np.ndarray):
    """
    Convert 0/1 bits back to normalized audio.
    """
    num_bits = (len(bits) // 8) * 8
    bits = bits[:num_bits].astype(np.uint8)

    byte_vals = np.packbits(bits, bitorder="big")

    if len(byte_vals) % 2 != 0:
        byte_vals = byte_vals[:-1]

    audio_i16 = byte_vals.astype("<u1").view("<i2")
    audio = audio_i16.astype(np.float64) / 32767.0
    audio = normalize_audio(audio)
    return audio


def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    return 2.0 * bits.astype(np.float64) - 1.0


def apply_channel(symbols: np.ndarray, ebn0_db: float, freq_offset: float, phase_offset: float):
    n = np.arange(len(symbols), dtype=np.float64)
    ebn0 = 10 ** (ebn0_db / 10.0)
    sigma = np.sqrt(1.0 / (2.0 * ebn0))

    tx = symbols * np.exp(1j * (2.0 * np.pi * freq_offset * n + phase_offset))
    noise = sigma * (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols)))
    rx = tx + noise
    return tx, rx, sigma


def estimate_frequency_bpsk(rx: np.ndarray) -> float:
    """
    Squaring removes BPSK data:
      rx^2 ≈ exp(j*2*(2π f n + φ))
    The phase increment is doubled, so divide by 4π to get cycles/sample.
    """
    rx_sq = rx ** 2
    phase_diff = np.angle(rx_sq[1:] * np.conj(rx_sq[:-1]))
    freq_est = np.mean(phase_diff) / (4.0 * np.pi)  # cycles/sample
    return freq_est


def estimate_coarse_phase(rx_corr: np.ndarray, symbols: np.ndarray, pilot_len: int) -> float:
    pilot_len = min(pilot_len, len(symbols))
    return np.angle(np.mean(rx_corr[:pilot_len] * np.conj(symbols[:pilot_len])))


def kalman_2state_track(rx_coarse: np.ndarray, symbols: np.ndarray, sigma: float, pilot_len: int):
    """
    State x = [phase, frequency]^T
    x(k+1) = [1 1; 0 1] x(k) + w
    Measurement: phase from rx_coarse * conj(symbol estimate)
    """
    N = len(rx_coarse)

    x = np.array([0.0, 0.0], dtype=np.float64)
    P = np.eye(2, dtype=np.float64)

    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]], dtype=np.float64)

    Q = np.array([[Q_PHASE, 0.0],
                  [0.0, Q_FREQ]], dtype=np.float64)

    R = R_SCALE * (sigma ** 2)

    s_hat = np.zeros(N, dtype=np.float64)
    s_hat[:min(pilot_len, N)] = symbols[:min(pilot_len, N)]

    phi_est = np.zeros(N, dtype=np.float64)
    omega_est = np.zeros(N, dtype=np.float64)
    decisions = np.zeros(N, dtype=np.uint8)
    rx_corrected = np.zeros(N, dtype=np.complex128)

    for k in range(N):
        # Prediction
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Symbol estimate
        if s_hat[k] == 0.0:
            if k > 0:
                s_hat[k] = 2.0 * float(decisions[k - 1]) - 1.0
            else:
                s_hat[k] = 1.0

        # Measurement
        z = np.angle(rx_coarse[k] * np.conj(s_hat[k]))

        # Wrapped innovation
        err = z - x_pred[0]
        err = (err + np.pi) % (2.0 * np.pi) - np.pi

        # Kalman update
        H = np.array([[1.0, 0.0]], dtype=np.float64)
        S = H @ P_pred @ H.T + R
        K = (P_pred @ H.T) / S

        x = x_pred + (K[:, 0] * err)
        P = (np.eye(2) - K @ H) @ P_pred

        phi_est[k] = x[0]
        omega_est[k] = x[1]

        # Fine correction
        corrected = rx_coarse[k] * np.exp(-1j * x[0])
        rx_corrected[k] = corrected

        # Hard decision
        decisions[k] = np.uint8(np.real(corrected) > 0)

        # This matches the MATLAB script you shared: always use the known symbol for next step
        if k + 1 < N:
            s_hat[k + 1] = symbols[k + 1]

    return decisions, rx_corrected, phi_est, omega_est


# =========================================================
# MAIN
# =========================================================
def main():
    np.random.seed(0)

    audio, Fs = sf.read(INPUT_WAV)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = normalize_audio(audio)

    play_audio(audio, Fs, "Playing original audio...")

    # Audio -> bits
    bits, _ = audio_to_bits(audio)

    # BPSK
    symbols = bpsk_modulate(bits)

    # Channel
    tx, rx, sigma = apply_channel(symbols, EBN0_DB, FREQ_OFFSET, PHASE_OFFSET)

    # =====================================================
    # Stage 1: Frequency estimation
    # =====================================================
    freq_est = estimate_frequency_bpsk(rx)
    print(f"Estimated frequency offset = {freq_est}")

    n = np.arange(len(symbols), dtype=np.float64)
    rx_freq_corrected = rx * np.exp(-1j * 2.0 * np.pi * freq_est * n)

    # =====================================================
    # Stage 2: Coarse phase estimation
    # =====================================================
    phi0 = estimate_coarse_phase(rx_freq_corrected, symbols, PILOT_LEN)
    print(f"Coarse phase estimate = {phi0}")

    rx_coarse = rx_freq_corrected * np.exp(-1j * phi0)

    temp_bits = (np.real(rx_coarse) > 0).astype(np.uint8)
    ber_coarse = np.mean(bits != temp_bits)
    print(f"BER after coarse only = {ber_coarse}")

    # =====================================================
    # Stage 3: 2-state Kalman filter
    # =====================================================
    decisions, rx_corrected, phi_est, omega_est = kalman_2state_track(
        rx_coarse, symbols, sigma, PILOT_LEN
    )

    # Phase ambiguity fix
    pilot_bits = bits[:min(PILOT_LEN, len(bits))]
    pilot_decisions = decisions[:min(PILOT_LEN, len(decisions))]

    if np.mean(pilot_bits != pilot_decisions) > 0.5:
        decisions = 1 - decisions
        rx_corrected = -rx_corrected
        print("Phase ambiguity corrected")

    bit_errors = np.sum(bits != decisions[:len(bits)])
    ber = bit_errors / len(bits)
    print(f"BER after 2-state Kalman = {ber}")

    # =====================================================
    # Bits -> audio
    # =====================================================
    num_bits = (len(decisions) // 8) * 8
    decisions_trim = decisions[:num_bits]

    audio_rx = bits_to_audio(decisions_trim)

    play_audio(audio_rx, Fs, "Playing recovered audio...")
    sf.write(OUTPUT_WAV, audio_rx, Fs)
    print(f"Recovered audio saved to {OUTPUT_WAV}")

    # =====================================================
    # Plots
    # =====================================================
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(audio)
    plt.title("Original Audio")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(audio_rx)
    plt.title("Recovered Audio")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "audio_waveforms.png"), dpi=200)

    plt.figure(figsize=(10, 4))
    plt.stem(bits[:200], basefmt=" ")
    plt.title("Original Bits (first 200)")
    plt.xlabel("Bit Index")
    plt.ylabel("Bit")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "original_bits.png"), dpi=200)

    plt.figure(figsize=(10, 4))
    plt.stem(decisions[:200], basefmt=" ")
    plt.title("Detected Bits (first 200)")
    plt.xlabel("Bit Index")
    plt.ylabel("Bit")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "detected_bits.png"), dpi=200)

    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(np.real(tx[:2000]))
    plt.title("Transmitted Signal (Real)")
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(np.real(rx[:2000]))
    plt.title("Received Signal (Real)")
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(np.real(rx_freq_corrected[:2000]))
    plt.title("After Frequency Correction (Real)")
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(np.real(rx_corrected[:2000]))
    plt.title("After Kalman Correction (Real)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "signal_waveforms.png"), dpi=200)

    plt.figure(figsize=(10, 5))
    plt.plot(phi_est)
    plt.title("Kalman Phase Estimate")
    plt.xlabel("Sample")
    plt.ylabel("Phase (rad)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "phase_estimate.png"), dpi=200)

    plt.figure(figsize=(10, 5))
    plt.plot(omega_est)
    plt.title("Kalman Frequency Estimate")
    plt.xlabel("Sample")
    plt.ylabel("Frequency (state units)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "frequency_estimate.png"), dpi=200)

    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(rx[:2000]), np.imag(rx[:2000]), s=4)
    plt.title("Before Correction")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "constellation_before.png"), dpi=200)

    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(rx_freq_corrected[:2000]), np.imag(rx_freq_corrected[:2000]), s=4)
    plt.title("After Frequency Correction")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "constellation_after_freq.png"), dpi=200)

    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(rx_corrected[:2000]), np.imag(rx_corrected[:2000]), s=4)
    plt.title("After Kalman Correction")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "constellation_after_kalman.png"), dpi=200)

    plt.show()


if __name__ == "__main__":
    main()
