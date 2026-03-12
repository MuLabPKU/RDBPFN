# This file contains the temporal vocabulary for the SCM

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import random

DEFAULT_NUM_POINTS = 1000


class ComponentType(Enum):
    TREND = "trend"
    SEASONALITY = "seasonality"
    SPIKES = "spikes"
    NOISE = "noise"


class TemporalCombinationConfigs:
    """Pre-defined temporal combination configurations."""

    CONFIGS = {
        "Trend-Dominated": {
            "probs": {
                ComponentType.TREND: 0.9,
                ComponentType.SEASONALITY: 0.2,
                ComponentType.SPIKES: 0.1,
                ComponentType.NOISE: 0.3,
            },
            "amplitudes": {
                ComponentType.TREND: 2.0,
                ComponentType.SEASONALITY: 0.3,
                ComponentType.SPIKES: 0.2,
                ComponentType.NOISE: 0.1,
            },
        },
        "Seasonality-Dominated": {
            "probs": {
                ComponentType.TREND: 0.3,
                ComponentType.SEASONALITY: 0.9,
                ComponentType.SPIKES: 0.1,
                ComponentType.NOISE: 0.3,
            },
            "amplitudes": {
                ComponentType.TREND: 0.5,
                ComponentType.SEASONALITY: 1.5,
                ComponentType.SPIKES: 0.2,
                ComponentType.NOISE: 0.1,
            },
        },
        "Spike-Dominated": {
            "probs": {
                ComponentType.TREND: 0.2,
                ComponentType.SEASONALITY: 0.3,
                ComponentType.SPIKES: 0.8,
                ComponentType.NOISE: 0.3,
            },
            "amplitudes": {
                ComponentType.TREND: 0.3,
                ComponentType.SEASONALITY: 0.3,
                ComponentType.SPIKES: 1.0,
                ComponentType.NOISE: 0.1,
            },
        },
        "Noise-Dominated": {
            "probs": {
                ComponentType.TREND: 0.2,
                ComponentType.SEASONALITY: 0.2,
                ComponentType.SPIKES: 0.1,
                ComponentType.NOISE: 0.9,
            },
            "amplitudes": {
                ComponentType.TREND: 0.3,
                ComponentType.SEASONALITY: 0.2,
                ComponentType.SPIKES: 0.1,
                ComponentType.NOISE: 0.8,
            },
        },
        "Balanced": {
            "probs": {
                ComponentType.TREND: 0.6,
                ComponentType.SEASONALITY: 0.6,
                ComponentType.SPIKES: 0.3,
                ComponentType.NOISE: 0.5,
            },
            "amplitudes": {
                ComponentType.TREND: 0.8,
                ComponentType.SEASONALITY: 0.8,
                ComponentType.SPIKES: 0.4,
                ComponentType.NOISE: 0.2,
            },
        },
        "Default": {
            "probs": {
                ComponentType.TREND: 0.8,
                ComponentType.SEASONALITY: 0.6,
                ComponentType.SPIKES: 0.3,
                ComponentType.NOISE: 0.9,
            },
            "amplitudes": {
                ComponentType.TREND: 1.0,
                ComponentType.SEASONALITY: 0.5,
                ComponentType.SPIKES: 0.3,
                ComponentType.NOISE: 0.1,
            },
        },
    }

    @classmethod
    def get_random_config(cls):
        """Randomly select a configuration from the predefined ones."""
        config_name = random.choice(list(cls.CONFIGS.keys()))
        return config_name, cls.CONFIGS[config_name]

    @classmethod
    def get_config(cls, name: str):
        """Get a specific configuration by name."""
        if name not in cls.CONFIGS:
            raise ValueError(
                f"Configuration '{name}' not found. Available: {list(cls.CONFIGS.keys())}"
            )
        return cls.CONFIGS[name]


class TemporalVocab:
    """
    Main temporal vocabulary that combines different temporal components.

    The vocabulary includes:
    - Trend: Linear, exponential, polynomial trends
    - Seasonality: Periodic patterns with various frequencies
    - Spikes: Sudden increases or decreases in intensity
    - Noise: Random variations
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize the temporal vocabulary.

        Parameters
        ----------
        device : str
            Device to store tensors on
        """
        self.device = device
        self.components = {}
        self.component_probs = {}
        self.component_amplitudes = {}

        # Initialize component vocabularies
        self.trend_vocab = TrendVocab(device=device)
        self.seasonality_vocab = SeasonalityVocab(device=device)
        self.spikes_vocab = SpikesVocab(device=device)
        self.noise_vocab = NoiseVocab(device=device)

        # Default probabilities and amplitudes
        self._init_default_params()

    def _init_default_params(self):
        """Initialize default probabilities and amplitudes for components."""
        # Randomly select one from pre-defined configurations
        config_name, config = TemporalCombinationConfigs.get_random_config()

        # print(f"Selected temporal configuration: {config_name}")

        self.component_probs = config["probs"].copy()
        self.component_amplitudes = config["amplitudes"].copy()
        self.config_name = config_name

    def init(
        self,
        component_probs: Dict[ComponentType, float] = None,
        component_amplitudes: Dict[ComponentType, float] = None,
    ):
        """
        Initialize the probabilities and amplitudes of different components.

        Parameters
        ----------
        component_probs : Dict[ComponentType, float], optional
            Probability of including each component type
        component_amplitudes : Dict[ComponentType, float], optional
            Amplitude/strength of each component type
        """
        if component_probs is not None:
            self.component_probs.update(component_probs)

        if component_amplitudes is not None:
            self.component_amplitudes.update(component_amplitudes)

    def generate(
        self,
        time_range: Tuple[float, float] = (0.0, 10.0),
        num_points: int = DEFAULT_NUM_POINTS,
    ) -> torch.Tensor:
        """
        Generate a temporal distribution by combining selected components.

        Parameters
        ----------
        time_range : Tuple[float, float]
            Start and end time for the distribution
        num_points : int
            Number of time points to generate

        Returns
        -------
        torch.Tensor
            Generated temporal distribution (time_points, intensity)
        """
        t_start, t_end = time_range
        t = torch.linspace(t_start, t_end, num_points, device=self.device)

        # Initialize with baseline intensity
        intensity = torch.ones_like(t, device=self.device, dtype=torch.float32)

        # Add components based on probabilities
        if random.random() < self.component_probs[ComponentType.TREND]:
            trend_component = self.trend_vocab.generate(t)
            intensity += (
                self.component_amplitudes[ComponentType.TREND] * trend_component
            )

        if random.random() < self.component_probs[ComponentType.SEASONALITY]:
            seasonal_component = self.seasonality_vocab.generate(t)
            intensity += (
                self.component_amplitudes[ComponentType.SEASONALITY]
                * seasonal_component
            )

        if random.random() < self.component_probs[ComponentType.SPIKES]:
            spikes_component = self.spikes_vocab.generate(t)
            intensity += (
                self.component_amplitudes[ComponentType.SPIKES] * spikes_component
            )

        if random.random() < self.component_probs[ComponentType.NOISE]:
            noise_component = self.noise_vocab.generate(t)
            intensity += (
                self.component_amplitudes[ComponentType.NOISE] * noise_component
            )

        # Ensure non-negative intensity
        intensity = torch.clamp(intensity, min=0.01)

        self.intensity = intensity

        return torch.stack([t, intensity], dim=1)

    def norm_intensity(self) -> torch.Tensor:
        """
        Normalize the intensity with mean 1, std 1, and clip to be non-negative and less than 10.
        """
        # Standardize to mean=0, std=1, then shift to mean=1
        intensity = (
            self.intensity - self.intensity.mean()
        ) / self.intensity.std() + 1.0
        # Fill nan with 1
        intensity = torch.where(
            torch.isnan(intensity), torch.ones_like(intensity), intensity
        )
        intensity = torch.clamp(intensity, min=0.01, max=10.0)
        self.intensity = intensity
        return

    @property
    def get_intensity(self) -> torch.Tensor:
        """
        Get the intensity of the temporal distribution.
        """
        return self.intensity

    def sample(
        self,
        num_samples: int,
        time_range: Tuple[float, float] = (0.0, 10.0),
        num_points: int = DEFAULT_NUM_POINTS,
        distribution: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample event times from the generated temporal distribution.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate
        time_range : Tuple[float, float]
            Time range for sampling
        distribution : torch.Tensor, optional
            Pre-generated distribution. If None, generates new one.

        Returns
        -------
        torch.Tensor
            Sampled event indices
        """
        if distribution is None:
            distribution = self.generate(time_range, num_points=num_points)

        # t_points = distribution[:, 0]
        intensity = distribution[:, 1]

        # Normalize intensity to create probability distribution
        probs = intensity / intensity.sum()

        # Sample indices based on probabilities
        sample_indices = torch.multinomial(probs, num_samples, replacement=True)

        # # Get corresponding time points
        # sampled_times = t_points[sample_indices]

        # # Sort samples
        # sampled_times = torch.sort(sampled_times)[0]

        return sample_indices

    def sample_time(
        self,
        num_samples: int,
        time_range: Tuple[float, float] = (0.0, 10.0),
        num_points: int = DEFAULT_NUM_POINTS,
        distribution: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample event times from the generated temporal distribution.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate
        time_range : Tuple[float, float]
            Time range for sampling
        distribution : torch.Tensor, optional
            Pre-generated distribution. If None, generates new one.

        Returns
        -------
        torch.Tensor
            Sampled event indices
        """
        if distribution is None:
            distribution = self.generate(time_range, num_points=num_points)

        t_points = distribution[:, 0]
        intensity = distribution[:, 1]

        # Normalize intensity to create probability distribution
        probs = intensity / intensity.sum()

        # Sample indices based on probabilities
        sample_indices = torch.multinomial(probs, num_samples, replacement=True)

        # Get corresponding time points
        sampled_times = t_points[sample_indices]

        # Sort samples
        sampled_times = torch.sort(sampled_times)[0]

        return sampled_times

    def retrieve(self, t: torch.Tensor) -> torch.Tensor:
        """
        Retrieve the intensity of the temporal distribution at a given time point.
        """
        return self.generate(t)

    def __len__(self):
        """Return total number of component patterns available."""
        return (
            len(self.trend_vocab)
            + len(self.seasonality_vocab)
            + len(self.spikes_vocab)
            + len(self.noise_vocab)
        )


class TrendVocab:
    """Vocabulary for trend components."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.patterns = [
            "linear_increasing",
            "linear_decreasing",
            "exponential_growth",
            "exponential_decay",
            "polynomial_quadratic",
            "polynomial_cubic",
            "logistic_growth",
            "power_law",
            "logarithmic",
            "constant",
        ]

    def init(self, pattern_probs: Dict[str, float] = None):
        """
        Initialize probabilities for different trend patterns.

        Parameters
        ----------
        pattern_probs : Dict[str, float], optional
            Probability of selecting each pattern
        """
        if pattern_probs is None:
            # Default uniform probabilities
            self.pattern_probs = {
                pattern: 1.0 / len(self.patterns) for pattern in self.patterns
            }
        else:
            self.pattern_probs = pattern_probs

    def generate(self, t: torch.Tensor) -> torch.Tensor:
        """
        Generate a trend component.

        Parameters
        ----------
        t : torch.Tensor
            Time points

        Returns
        -------
        torch.Tensor
            Trend values
        """
        if not hasattr(self, "pattern_probs"):
            self.init()

        # Select pattern based on probabilities
        pattern_names = list(self.pattern_probs.keys())
        pattern_weights = list(self.pattern_probs.values())
        selected_pattern = np.random.choice(
            pattern_names, p=np.array(pattern_weights) / sum(pattern_weights)
        )

        # Generate trend based on selected pattern
        if selected_pattern == "linear_increasing":
            slope = random.uniform(0.1, 1.0)
            return slope * t

        elif selected_pattern == "linear_decreasing":
            slope = random.uniform(-1.0, -0.1)
            return slope * t

        elif selected_pattern == "exponential_growth":
            rate = random.uniform(0.1, 0.5)
            return torch.exp(rate * t) - 1

        elif selected_pattern == "exponential_decay":
            rate = random.uniform(0.1, 0.5)
            return torch.exp(-rate * t)

        elif selected_pattern == "polynomial_quadratic":
            a = random.uniform(-0.1, 0.1)
            b = random.uniform(-0.5, 0.5)
            return a * t**2 + b * t

        elif selected_pattern == "polynomial_cubic":
            a = random.uniform(-0.01, 0.01)
            b = random.uniform(-0.1, 0.1)
            c = random.uniform(-0.5, 0.5)
            return a * t**3 + b * t**2 + c * t

        elif selected_pattern == "logistic_growth":
            k = random.uniform(1.0, 5.0)  # carrying capacity
            r = random.uniform(0.1, 1.0)  # growth rate
            t0 = random.uniform(0.2, 0.8) * t.max()  # inflection point
            return k / (1 + torch.exp(-r * (t - t0)))

        elif selected_pattern == "power_law":
            alpha = random.uniform(0.5, 2.0)
            return torch.pow(t + 1, alpha) - 1

        elif selected_pattern == "logarithmic":
            scale = random.uniform(0.5, 2.0)
            return scale * torch.log(t + 1)

        else:  # constant
            return torch.zeros_like(t)

    def __len__(self):
        return len(self.patterns)


class SeasonalityVocab:
    """Vocabulary for seasonality components."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.patterns = [
            "sinusoidal",
            "cosine",
            "sawtooth",
            "square_wave",
            "triangle_wave",
            "multi_frequency",
            "damped_oscillation",
            "frequency_modulation",
            "amplitude_modulation",
            "harmonic_series",
        ]

    def init(self, pattern_probs: Dict[str, float] = None):
        """Initialize probabilities for different seasonality patterns."""
        if pattern_probs is None:
            self.pattern_probs = {
                pattern: 1.0 / len(self.patterns) for pattern in self.patterns
            }
        else:
            self.pattern_probs = pattern_probs

    def generate(self, t: torch.Tensor) -> torch.Tensor:
        """Generate a seasonality component."""
        if not hasattr(self, "pattern_probs"):
            self.init()

        pattern_names = list(self.pattern_probs.keys())
        pattern_weights = list(self.pattern_probs.values())
        selected_pattern = np.random.choice(
            pattern_names, p=np.array(pattern_weights) / sum(pattern_weights)
        )

        # Base frequency and parameters
        frequency = random.uniform(0.5, 3.0)
        phase = random.uniform(0, 2 * np.pi)

        if selected_pattern == "sinusoidal":
            return torch.sin(2 * np.pi * frequency * t + phase)

        elif selected_pattern == "cosine":
            return torch.cos(2 * np.pi * frequency * t + phase)

        elif selected_pattern == "sawtooth":
            return (
                2
                * (
                    frequency * t
                    + phase / (2 * np.pi)
                    - torch.floor(frequency * t + phase / (2 * np.pi))
                )
                - 1
            )

        elif selected_pattern == "square_wave":
            return torch.sign(torch.sin(2 * np.pi * frequency * t + phase))

        elif selected_pattern == "triangle_wave":
            return (
                2 / np.pi * torch.arcsin(torch.sin(2 * np.pi * frequency * t + phase))
            )

        elif selected_pattern == "multi_frequency":
            # Combine multiple frequencies
            result = torch.zeros_like(t)
            for i in range(random.randint(2, 4)):
                freq_i = frequency * (i + 1)
                weight_i = 1.0 / (i + 1)
                result += weight_i * torch.sin(2 * np.pi * freq_i * t + phase)
            return result

        elif selected_pattern == "damped_oscillation":
            decay_rate = random.uniform(0.1, 1.0)
            return torch.exp(-decay_rate * t) * torch.sin(
                2 * np.pi * frequency * t + phase
            )

        elif selected_pattern == "frequency_modulation":
            mod_freq = frequency / random.uniform(2, 5)
            mod_depth = random.uniform(0.5, 2.0)
            instantaneous_freq = frequency + mod_depth * torch.sin(
                2 * np.pi * mod_freq * t
            )
            return torch.sin(
                2 * np.pi * torch.cumsum(instantaneous_freq, dim=0) * (t[1] - t[0])
                + phase
            )

        elif selected_pattern == "amplitude_modulation":
            mod_freq = frequency / random.uniform(3, 8)
            mod_depth = random.uniform(0.3, 0.8)
            envelope = 1 + mod_depth * torch.sin(2 * np.pi * mod_freq * t)
            return envelope * torch.sin(2 * np.pi * frequency * t + phase)

        else:  # harmonic_series
            result = torch.zeros_like(t)
            for n in range(1, random.randint(4, 7)):
                harmonic_weight = 1.0 / n
                result += harmonic_weight * torch.sin(
                    2 * np.pi * n * frequency * t + phase
                )
            return result

    def __len__(self):
        return len(self.patterns)


class SpikesVocab:
    """Vocabulary for spike components."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.patterns = [
            "gaussian_spikes",
            "exponential_spikes",
            "rectangular_spikes",
            "triangular_spikes",
            "clustered_spikes",
            "periodic_spikes",
            "random_spikes",
            "burst_spikes",
        ]

    def init(self, pattern_probs: Dict[str, float] = None):
        """Initialize probabilities for different spike patterns."""
        if pattern_probs is None:
            self.pattern_probs = {
                pattern: 1.0 / len(self.patterns) for pattern in self.patterns
            }
        else:
            self.pattern_probs = pattern_probs

    def generate(self, t: torch.Tensor) -> torch.Tensor:
        """Generate a spikes component."""
        if not hasattr(self, "pattern_probs"):
            self.init()

        pattern_names = list(self.pattern_probs.keys())
        pattern_weights = list(self.pattern_probs.values())
        selected_pattern = np.random.choice(
            pattern_names, p=np.array(pattern_weights) / sum(pattern_weights)
        )

        result = torch.zeros_like(t)
        num_spikes = random.randint(1, 5)

        for _ in range(num_spikes):
            # Random spike location and parameters
            spike_center = random.uniform(t.min().item(), t.max().item())
            spike_width = random.uniform(0.1, 0.5) * (t.max() - t.min()) / 10
            spike_amplitude = random.uniform(0.5, 2.0)

            if selected_pattern == "gaussian_spikes":
                spike = spike_amplitude * torch.exp(
                    -0.5 * ((t - spike_center) / spike_width) ** 2
                )

            elif selected_pattern == "exponential_spikes":
                spike = spike_amplitude * torch.exp(
                    -torch.abs(t - spike_center) / spike_width
                )

            elif selected_pattern == "rectangular_spikes":
                spike = torch.where(
                    torch.abs(t - spike_center) <= spike_width,
                    spike_amplitude,
                    torch.tensor(0.0),
                )

            elif selected_pattern == "triangular_spikes":
                distance = torch.abs(t - spike_center)
                spike = torch.where(
                    distance <= spike_width,
                    spike_amplitude * (1 - distance / spike_width),
                    torch.tensor(0.0),
                )

            else:  # Default to Gaussian
                spike = spike_amplitude * torch.exp(
                    -0.5 * ((t - spike_center) / spike_width) ** 2
                )

            result += spike

        return result

    def __len__(self):
        return len(self.patterns)


class NoiseVocab:
    """Vocabulary for noise components."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.patterns = [
            "white_noise",
            "colored_noise",
            "brownian_motion",
            "ou_process",  # Ornstein-Uhlenbeck
            "fractional_brownian",
            "poisson_noise",
            "uniform_noise",
        ]

    def init(self, pattern_probs: Dict[str, float] = None):
        """Initialize probabilities for different noise patterns."""
        if pattern_probs is None:
            self.pattern_probs = {
                pattern: 1.0 / len(self.patterns) for pattern in self.patterns
            }
        else:
            self.pattern_probs = pattern_probs

    def generate(self, t: torch.Tensor) -> torch.Tensor:
        """Generate a noise component."""
        if not hasattr(self, "pattern_probs"):
            self.init()

        pattern_names = list(self.pattern_probs.keys())
        pattern_weights = list(self.pattern_probs.values())
        selected_pattern = np.random.choice(
            pattern_names, p=np.array(pattern_weights) / sum(pattern_weights)
        )

        if selected_pattern == "white_noise":
            return torch.randn_like(t, device=self.device)

        elif selected_pattern == "colored_noise":
            # Simple colored noise (low-pass filtered white noise)
            white = torch.randn_like(t, device=self.device)
            alpha = random.uniform(0.1, 0.9)
            colored = torch.zeros_like(white)
            colored[0] = white[0]
            for i in range(1, len(white)):
                colored[i] = alpha * colored[i - 1] + (1 - alpha) * white[i]
            return colored

        elif selected_pattern == "brownian_motion":
            increments = torch.randn_like(t, device=self.device)
            return torch.cumsum(increments, dim=0)

        elif selected_pattern == "ou_process":
            # Ornstein-Uhlenbeck process
            theta = random.uniform(0.1, 1.0)  # mean reversion rate
            sigma = random.uniform(0.5, 1.5)  # volatility
            dt = (t[1] - t[0]).item() if len(t) > 1 else 0.01

            ou = torch.zeros_like(t)
            for i in range(1, len(t)):
                dW = torch.randn(1, device=self.device) * torch.sqrt(torch.tensor(dt))
                ou[i] = ou[i - 1] + theta * (0 - ou[i - 1]) * dt + sigma * dW
            return ou

        elif selected_pattern == "uniform_noise":
            return torch.rand_like(t, device=self.device) * 2 - 1  # Range [-1, 1]

        else:  # Default to white noise
            return torch.randn_like(t, device=self.device)

    def __len__(self):
        return len(self.patterns)
