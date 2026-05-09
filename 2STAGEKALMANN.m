clc;
clear;
close all;

%% =========================================================
%% READ AUDIO
%% =========================================================
[audio, Fs] = audioread('input.wav');

% Convert stereo to mono
if size(audio,2) > 1
    audio = mean(audio,2);
end

% Normalize
audio = audio / max(abs(audio));

disp('Playing original audio...'); %[output:942ee28e]
soundsc(audio, Fs);
pause(length(audio)/Fs + 1);

%% =========================================================
%% AUDIO -> BITS
%% =========================================================
audio_i16 = int16(audio * 32767);

bytes = typecast(audio_i16(:), 'uint8');

bitMat = zeros(length(bytes),8);

for i = 1:8
    bitMat(:,i) = bitget(bytes,9-i);
end

bits = reshape(bitMat.',1,[]);

%% =========================================================
%% BPSK MODULATION
%% =========================================================
symbols = 2*double(bits) - 1;

%% =========================================================
%% CHANNEL
%% =========================================================
EbN0_dB = 10;

EbN0 = 10^(EbN0_dB/10);

sigma = sqrt(1/(2*EbN0));

phase_offset = pi/4;

freq_offset = 0.0001;

N = length(symbols);

n = 0:N-1;

% Channel impairments
tx = symbols .* exp(1j*(2*pi*freq_offset*n + phase_offset));

% AWGN
noise = sigma * (randn(size(tx)) + 1j*randn(size(tx)));

rx = tx + noise;

%% =========================================================
%% STAGE 1 : COARSE FREQUENCY ESTIMATION
%% =========================================================

% BPSK squaring trick
rx_sq = rx.^2;

% Estimate phase difference
phase_diff = angle(rx_sq(2:end) .* conj(rx_sq(1:end-1)));

% Average estimate
freq_est = mean(phase_diff)/(4*pi);

disp(['Estimated frequency offset = ', num2str(freq_est)]) %[output:8fd77c39]

% Frequency correction
rx_freq_corrected = rx .* exp(-1j*2*pi*freq_est*n);

%% =========================================================
%% STAGE 2 : COARSE PHASE ESTIMATION
%% =========================================================

pilot_len = 100;

phi0 = angle(mean( ...
       rx_freq_corrected(1:pilot_len) .* ...
       conj(symbols(1:pilot_len)) ));

disp(['Coarse phase estimate = ', num2str(phi0)]) %[output:62c73dcd]

% Phase correction
rx_coarse = rx_freq_corrected .* exp(-1j*phi0);

% BER after coarse correction
temp_bits = real(rx_coarse) > 0;

ber_coarse = mean(bits ~= temp_bits);

disp(['BER after coarse only = ', num2str(ber_coarse)]) %[output:48d78916]

%% =========================================================
%% STAGE 3 : 2-STATE KALMAN FILTER
%% =========================================================

% State:
% x(1) = phase
% x(2) = frequency

x = [0; 0];

P = eye(2);

F = [1 1;
     0 1];

Q = [1e-5 0;
     0     1e-7];

R = 5*sigma^2;

phi_est = zeros(1,N);

omega_est = zeros(1,N);

decisions = zeros(1,N);

rx_corrected = zeros(1,N);

%% Pilot-assisted initialization

s_hat = zeros(1,N);

s_hat(1:pilot_len) = symbols(1:pilot_len);

%% =========================================================
%% KALMAN LOOP
%% =========================================================

for k = 1:N

    %% Prediction
    x_pred = F * x;

    P_pred = F * P * F' + Q;

    %% Symbol estimate
    if s_hat(k) == 0

        if k > 1
            s_hat(k) = 2*decisions(k-1) - 1;
        else
            s_hat(k) = 1;
        end

    end

    %% Measurement

    z = angle(rx_coarse(k) * conj(s_hat(k)));

    %% Wrapped innovation

    err = z - x_pred(1);

    err = mod(err + pi, 2*pi) - pi;

    %% Measurement matrix

    H = [1 0];

    %% Kalman gain

    K = P_pred * H' / (H * P_pred * H' + R);

    %% Update

    x = x_pred + K * err;

    P = (eye(2) - K * H) * P_pred;

    %% Save estimates

    phi_est(k) = x(1);

    omega_est(k) = x(2);

    %% Fine correction

    corrected = rx_coarse(k) * exp(-1j*x(1));

    rx_corrected(k) = corrected;

    %% Hard decision

    decisions(k) = real(corrected) > 0;

    %% Decision-directed update

    if k < N
    s_hat(k+1) = symbols(k+1);
    end

end

%% =========================================================
%% PHASE AMBIGUITY FIX
%% =========================================================

pilot_bits = bits(1:pilot_len);

pilot_decisions = decisions(1:pilot_len);

if mean(pilot_bits ~= pilot_decisions) > 0.5

    decisions = ~decisions;

    rx_corrected = -rx_corrected;

    disp('Phase ambiguity corrected')

end

%% =========================================================
%% FINAL BER
%% =========================================================

bit_errors = sum(bits ~= decisions);

BER = bit_errors / length(bits);

disp(['BER after 2-state Kalman = ', num2str(BER)]) %[output:2bc19581]

%% =========================================================
%% BITS -> AUDIO
%% =========================================================

num_bits = floor(length(decisions)/8)*8;

decisions = decisions(1:num_bits);

rx_bitMat = reshape(decisions,8,[]).';

weights = 2.^(7:-1:0);

rx_bytes = uint8(double(rx_bitMat) * weights.');

audio_i16_rx = typecast(rx_bytes,'int16');

audio_rx = double(audio_i16_rx)/32767;

audio_rx = audio_rx / max(abs(audio_rx));

disp('Playing recovered audio...'); %[output:9e4644ad]

soundsc(audio_rx, Fs);

pause(length(audio_rx)/Fs + 1);

audiowrite('recovered_audio.wav', audio_rx, Fs);

%% =========================================================
%% PLOTS
%% =========================================================

% ---------------------------------------------------------
% Audio waveforms
% ---------------------------------------------------------

figure;

subplot(2,1,1);

plot(audio);

title('Original Audio');

xlabel('Sample');

ylabel('Amplitude');

grid on;

subplot(2,1,2);

plot(audio_rx);

title('Recovered Audio');

xlabel('Sample');

ylabel('Amplitude');

grid on;

% ---------------------------------------------------------
% Bit streams
% ---------------------------------------------------------

figure;

stem(bits(1:200),'filled');

title('Original Bits');

xlabel('Bit Index');

ylabel('Bit');

grid on;

figure;

stem(decisions(1:200),'filled');

title('Detected Bits');

xlabel('Bit Index');

ylabel('Bit');

grid on;

% ---------------------------------------------------------
% BPSK symbols
% ---------------------------------------------------------

figure;

stem(symbols(1:200),'filled');

title('BPSK Symbols');

xlabel('Symbol Index');

ylabel('Amplitude');

grid on;

% ---------------------------------------------------------
% Signal waveforms
% ---------------------------------------------------------

figure;

subplot(4,1,1);

plot(real(tx(1:2000)));

title('Transmitted Signal');

grid on;

subplot(4,1,2);

plot(real(rx(1:2000)));

title('Received Signal');

grid on;

subplot(4,1,3);

plot(real(rx_freq_corrected(1:2000)));

title('After Frequency Correction');

grid on;

subplot(4,1,4);

plot(real(rx_corrected(1:2000)));

title('After Kalman Correction');

grid on;

% ---------------------------------------------------------
% Phase estimate
% ---------------------------------------------------------

figure;

plot(phi_est);

title('Kalman Phase Estimate');

xlabel('Sample');

ylabel('Phase');

grid on;

% ---------------------------------------------------------
% Frequency estimate
% ---------------------------------------------------------

figure;

plot(omega_est);

title('Kalman Frequency Estimate');

xlabel('Sample');

ylabel('Frequency');

grid on;

% ---------------------------------------------------------
% Constellations
% ---------------------------------------------------------

figure;

scatter(real(rx(1:2000)), imag(rx(1:2000)), '.');

title('Before Correction');

xlabel('In-phase');

ylabel('Quadrature');

grid on;

figure;

scatter(real(rx_freq_corrected(1:2000)), ...
        imag(rx_freq_corrected(1:2000)), '.');

title('After Frequency Correction');

xlabel('In-phase');

ylabel('Quadrature');

grid on;

figure;

scatter(real(rx_corrected(1:2000)), ...
        imag(rx_corrected(1:2000)), '.');

title('After Kalman Correction');

xlabel('In-phase');

ylabel('Quadrature');

grid on;

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright"}
%---
%[output:942ee28e]
%   data: {"dataType":"text","outputData":{"text":"Playing original audio...\n","truncated":false}}
%---
%[output:8fd77c39]
%   data: {"dataType":"text","outputData":{"text":"Estimated frequency offset = 9.8816e-05\n","truncated":false}}
%---
%[output:62c73dcd]
%   data: {"dataType":"text","outputData":{"text":"Coarse phase estimate = 0.81665\n","truncated":false}}
%---
%[output:48d78916]
%   data: {"dataType":"text","outputData":{"text":"BER after coarse only = 0.50368\n","truncated":false}}
%---
%[output:2bc19581]
%   data: {"dataType":"text","outputData":{"text":"BER after 2-state Kalman = 4.0539e-06\n","truncated":false}}
%---
%[output:9e4644ad]
%   data: {"dataType":"text","outputData":{"text":"Playing recovered audio...\n","truncated":false}}
%---
