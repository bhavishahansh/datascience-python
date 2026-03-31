from abc import ABC, abstractmethod

class PricingStrategy(ABC):

    @abstractmethod
    def calculate_fare(self, distance_km: float) -> float:
        pass


class StandardPricing(PricingStrategy):

    def calculate_fare(self, distance_km: float) -> float:
        base = 50
        return base + distance_km * 10


class SurgePricing(PricingStrategy):

    def calculate_fare(self, distance_km: float) -> float:
        base = 50
        return base + distance_km * 10 * 1.5
