import pandas as pd
import numpy as np
import time

def configure_child_dataset(i, n_samples):

    # Creazione dataset Rete Bayesiana CHILD

    start_time_child_config = time.time()

    data = pd.DataFrame(columns=["BirthAsphyxia", "Disease", "Sick", "DuctFlow", "CardiacMixing", "LungParench", "LungFlow", "LVH", 
                                "Age", "Grunting", "HypDistrib", "HypoxiaInO2", "CO2", "ChestXray", "LVHreport", "GruntingReport",
                                "LowerBodyO2", "RUQO2", "CO2Report", "XrayReport" ])
    # Livello 1 - Nodo iniziale
    data['BirthAsphyxia'] = np.random.choice([1, 0], size=n_samples, p=[0.1, 0.9])

    data["HypDistrib"] = np.where(data["BirthAsphyxia"] == 1, np.random.choice([1, 0], size=n_samples, p=[0.1, 0.9]), 0)

    # Disease depends on BirthAsphyxia
    data["Disease"] = np.where(data["BirthAsphyxia"] == 1, 
                            np.random.choice([5, 4, 3, 2, 1, 0], size=n_samples, p=[0.20, 0.30, 0.25, 0.15, 0.05, 0.05]), 
                            np.random.choice([5, 4, 3, 2, 1, 0], size=n_samples, p=[0.03061224, 0.33673469, 0.29591837, 0.23469388, 0.05102041, 0.05102041]))


    # LVH depends on Disease
    data["LVH"] = np.select(
        [
            data["Disease"] == 5,  # PFC
            data["Disease"] == 4,  # TGA
            data["Disease"] == 3,  # Fallot
            data["Disease"] == 2,  # PAIVS
            data["Disease"] == 1,  # TAPVD
            data["Disease"] == 0,  # Lung
        ],
        [
            np.random.choice([1, 0], size=n_samples, p=[0.1, 0.9]),
            np.random.choice([1, 0], size=n_samples, p=[0.1, 0.9]),
            np.random.choice([1, 0], size=n_samples, p=[0.1, 0.9]),
            np.random.choice([1, 0], size=n_samples, p=[0.9, 0.1]),
            np.random.choice([1, 0], size=n_samples, p=[0.05, 0.95]),
            np.random.choice([1, 0], size=n_samples, p=[0.1, 0.9]),
        ],
        default=0
    )

    # LVHreport depends on LVH
    data["LVHreport"] = np.where(data["LVH"] == 1, 
                                np.random.choice([1, 0], size=n_samples, p=[0.9, 0.1]), 
                                np.random.choice([1, 0], size=n_samples, p=[0.05, 0.95]))

    # DuctFlow depends on Disease
    data["DuctFlow"] = np.select(
        [
            data["Disease"] == 5,  # PFC
            data["Disease"] == 4,  # TGA
            data["Disease"] == 3,  # Fallot
            data["Disease"] == 2,  # PAIVS
            data["Disease"] == 1,  # TAPVD
            data["Disease"] == 0,  # Lung
        ],
        [
            np.random.choice([2, 1, 0], size=n_samples, p=[0.15, 0.05, 0.80]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.1, 0.8, 0.1]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.8, 0.2, 0.0]),
            np.random.choice([2, 1, 0], size=n_samples, p=[1.0, 0.0, 0.0]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.33, 0.33, 0.34]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.2, 0.4, 0.4]),
        ],
        default=0
    )

    # CardiacMixing depends on Disease
    data["CardiacMixing"] = np.select(
        [
            data["Disease"] == 5,  # PFC
            data["Disease"] == 4,  # TGA
            data["Disease"] == 3,  # Fallot
            data["Disease"] == 2,  # PAIVS
            data["Disease"] == 1,  # TAPVD
            data["Disease"] == 0,  # Lung
        ],
        [
            np.random.choice([3, 2, 1, 0], size=n_samples, p=[0.40, 0.43, 0.15, 0.02]),
            np.random.choice([3, 2, 1, 0], size=n_samples, p=[0.02, 0.09, 0.09, 0.80]),
            np.random.choice([3, 2, 1, 0], size=n_samples, p=[0.02, 0.16, 0.80, 0.02]),
            np.random.choice([3, 2, 1, 0], size=n_samples, p=[0.01, 0.02, 0.95, 0.02]),
            np.random.choice([3, 2, 1, 0], size=n_samples, p=[0.01, 0.03, 0.95, 0.01]),
            np.random.choice([3, 2, 1, 0], size=n_samples, p=[0.40, 0.53, 0.05, 0.02]),
        ],
        default=0
    )

    # HypDistrib depends on DuctFlow and CardiacMixing
    data["HypDistrib"] = np.select(
        [
            (data["DuctFlow"] == 2) & (data["CardiacMixing"] == 3),  # Lt_to_Rt, None
            (data["DuctFlow"] == 1) & (data["CardiacMixing"] == 3),  # None, None
            (data["DuctFlow"] == 0) & (data["CardiacMixing"] == 3),  # Rt_to_Lt, None
            (data["DuctFlow"] == 2) & (data["CardiacMixing"] == 2),  # Lt_to_Rt, Mild
            (data["DuctFlow"] == 1) & (data["CardiacMixing"] == 2),  # None, Mild
            (data["DuctFlow"] == 0) & (data["CardiacMixing"] == 2),  # Rt_to_Lt, Mild
            (data["DuctFlow"] == 2) & (data["CardiacMixing"] == 1),  # Lt_to_Rt, Complete
            (data["DuctFlow"] == 1) & (data["CardiacMixing"] == 1),  # None, Complete
            (data["DuctFlow"] == 0) & (data["CardiacMixing"] == 1),  # Rt_to_Lt, Complete
            (data["DuctFlow"] == 2) & (data["CardiacMixing"] == 0),  # Lt_to_Rt, Transp.
            (data["DuctFlow"] == 1) & (data["CardiacMixing"] == 0),  # None, Transp.
            (data["DuctFlow"] == 0) & (data["CardiacMixing"] == 0),  # Rt_to_Lt, Transp.
        ],
        [
            np.random.choice([1, 0], size=n_samples, p=[0.95, 0.05]),
            np.random.choice([1, 0], size=n_samples, p=[0.95, 0.05]),
            np.random.choice([1, 0], size=n_samples, p=[0.05, 0.95]),
            np.random.choice([1, 0], size=n_samples, p=[0.95, 0.05]),
            np.random.choice([1, 0], size=n_samples, p=[0.95, 0.05]),
            np.random.choice([1, 0], size=n_samples, p=[0.5, 0.5]),
            np.random.choice([1, 0], size=n_samples, p=[0.95, 0.05]),
            np.random.choice([1, 0], size=n_samples, p=[0.95, 0.05]),
            np.random.choice([1, 0], size=n_samples, p=[0.95, 0.05]),
            np.random.choice([1, 0], size=n_samples, p=[0.95, 0.05]),
            np.random.choice([1, 0], size=n_samples, p=[0.95, 0.05]),
            np.random.choice([1, 0], size=n_samples, p=[0.5, 0.5]),
        ],
        default=0
    )

    # LungParench depends on Disease
    data["LungParench"] = np.select(
        [
            data["Disease"] == 5,  # PFC
            data["Disease"] == 4,  # TGA
            data["Disease"] == 3,  # Fallot
            data["Disease"] == 2,  # PAIVS
            data["Disease"] == 1,  # TAPVD
            data["Disease"] == 0,  # Lung
        ],
        [
            np.random.choice([2, 1, 0], size=n_samples, p=[0.6, 0.1, 0.3]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.80, 0.05, 0.15]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.80, 0.05, 0.15]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.80, 0.05, 0.15]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.1, 0.6, 0.3]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.03, 0.25, 0.72]),
        ],
        default=0
    )

    # HypoxiaInO2 depends on CardiacMixing and LungParench
    data["HypoxiaInO2"] = np.select(
        [
            (data["CardiacMixing"] == 3) & (data["LungParench"] == 2),  # None, Normal
            (data["CardiacMixing"] == 2) & (data["LungParench"] == 2),  # Mild, Normal
            (data["CardiacMixing"] == 1) & (data["LungParench"] == 2),  # Complete, Normal
            (data["CardiacMixing"] == 0) & (data["LungParench"] == 2),  # Transp., Normal
            (data["CardiacMixing"] == 3) & (data["LungParench"] == 1),  # None, Congested
            (data["CardiacMixing"] == 2) & (data["LungParench"] == 1),  # Mild, Congested
            (data["CardiacMixing"] == 1) & (data["LungParench"] == 1),  # Complete, Congested
            (data["CardiacMixing"] == 0) & (data["LungParench"] == 1),  # Transp., Congested
            (data["CardiacMixing"] == 3) & (data["LungParench"] == 0),  # None, Abnormal
            (data["CardiacMixing"] == 2) & (data["LungParench"] == 0),  # Mild, Abnormal
            (data["CardiacMixing"] == 1) & (data["LungParench"] == 0),  # Complete, Abnormal
            (data["CardiacMixing"] == 0) & (data["LungParench"] == 0),  # Transp., Abnormal
        ],
        [
            np.random.choice([2, 1, 0], size=n_samples, p=[0.93, 0.05, 0.02]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.1, 0.8, 0.1]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.1, 0.7, 0.2]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.02, 0.18, 0.80]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.15, 0.80, 0.05]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.10, 0.75, 0.15]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.05, 0.65, 0.30]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.1, 0.3, 0.6]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.7, 0.2, 0.1]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.10, 0.65, 0.25]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.1, 0.5, 0.4]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.02, 0.18, 0.80]),
        ],
        default=0
    )

    # RUQO2 depends on HypoxiaInO2
    data["RUQO2"] = np.select(
        [
            data["HypoxiaInO2"] == 2,  # Mild
            data["HypoxiaInO2"] == 1,  # Moderate
            data["HypoxiaInO2"] == 0,  # Severe
        ],
        [
            np.random.choice([2, 1, 0], size=n_samples, p=[0.1, 0.3, 0.6]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.3, 0.6, 0.1]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.5, 0.4, 0.1]),
        ],
        default=0
    )

    # LowerBodyO2 depends on HypDistrib and HypoxiaInO2
    data["LowerBodyO2"] = np.select(
        [
            (data["HypDistrib"] == 1) & (data["HypoxiaInO2"] == 2),  # Equal, Mild
            (data["HypDistrib"] == 0) & (data["HypoxiaInO2"] == 2),  # Unequal, Mild
            (data["HypDistrib"] == 1) & (data["HypoxiaInO2"] == 1),  # Equal, Moderate
            (data["HypDistrib"] == 0) & (data["HypoxiaInO2"] == 1),  # Unequal, Moderate
            (data["HypDistrib"] == 1) & (data["HypoxiaInO2"] == 0),  # Equal, Severe
            (data["HypDistrib"] == 0) & (data["HypoxiaInO2"] == 0),  # Unequal, Severe
        ],
        [
            np.random.choice([2, 1, 0], size=n_samples, p=[0.1, 0.3, 0.6]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.4, 0.5, 0.1]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.3, 0.6, 0.1]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.50, 0.45, 0.05]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.5, 0.4, 0.1]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.60, 0.35, 0.05]),
        ],
        default=0
    )

    # CO2 depends on LungParench
    data["CO2"] = np.select(
        [
            data["LungParench"] == 2,  # Normal
            data["LungParench"] == 1,  # Congested
            data["LungParench"] == 0,  # Abnormal
        ],
        [
            np.random.choice([2, 1, 0], size=n_samples, p=[0.8, 0.1, 0.1]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.65, 0.05, 0.30]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.45, 0.05, 0.50]),
        ],
        default=0
    )

    # CO2Report depends on CO2
    data["CO2Report"] = np.select(
        [
            data["CO2"] == 2,  # Normal
            data["CO2"] == 1,  # Low
            data["CO2"] == 0,  # High
        ],
        [
            np.random.choice([1, 0], size=n_samples, p=[0.9, 0.1]),
            np.random.choice([1, 0], size=n_samples, p=[0.9, 0.1]),
            np.random.choice([1, 0], size=n_samples, p=[0.1, 0.9]),
        ],
        default=0
    )

    # LungFlow depends on Disease
    data["LungFlow"] = np.select(
        [
            data["Disease"] == 5,  # PFC
            data["Disease"] == 4,  # TGA
            data["Disease"] == 3,  # Fallot
            data["Disease"] == 2,  # PAIVS
            data["Disease"] == 1,  # TAPVD
            data["Disease"] == 0,  # Lung
        ],
        [
            np.random.choice([2, 1, 0], size=n_samples, p=[0.30, 0.65, 0.05]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.20, 0.05, 0.75]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.15, 0.80, 0.05]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.10, 0.85, 0.05]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.30, 0.10, 0.60]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.70, 0.10, 0.20]),
        ],
        default=0
    )

    # Sick depends on Disease
    data["Sick"] = np.select(
        [
            data["Disease"] == 5,  # PFC
            data["Disease"] == 4,  # TGA
            data["Disease"] == 3,  # Fallot
            data["Disease"] == 2,  # PAIVS
            data["Disease"] == 1,  # TAPVD
            data["Disease"] == 0,  # Lung
        ],
        [
            np.random.choice([1, 0], size=n_samples, p=[0.4, 0.6]),
            np.random.choice([1, 0], size=n_samples, p=[0.3, 0.7]),
            np.random.choice([1, 0], size=n_samples, p=[0.2, 0.8]),
            np.random.choice([1, 0], size=n_samples, p=[0.3, 0.7]),
            np.random.choice([1, 0], size=n_samples, p=[0.7, 0.3]),
            np.random.choice([1, 0], size=n_samples, p=[0.7, 0.3]),
        ],
        default=0
    )

    # Age depends on Disease and Sick
    data["Age"] = np.select(
        [
            (data["Disease"] == 5) & (data["Sick"] == 1),  # PFC, yes
            (data["Disease"] == 4) & (data["Sick"] == 1),  # TGA, yes
            (data["Disease"] == 3) & (data["Sick"] == 1),  # Fallot, yes
            (data["Disease"] == 2) & (data["Sick"] == 1),  # PAIVS, yes
            (data["Disease"] == 1) & (data["Sick"] == 1),  # TAPVD, yes
            (data["Disease"] == 0) & (data["Sick"] == 1),  # Lung, yes
            (data["Disease"] == 5) & (data["Sick"] == 0),  # PFC, no
            (data["Disease"] == 4) & (data["Sick"] == 0),  # TGA, no
            (data["Disease"] == 3) & (data["Sick"] == 0),  # Fallot, no
            (data["Disease"] == 2) & (data["Sick"] == 0),  # PAIVS, no
            (data["Disease"] == 1) & (data["Sick"] == 0),  # TAPVD, no
            (data["Disease"] == 0) & (data["Sick"] == 0),  # Lung, no
        ],
        [
            np.random.choice([2, 1, 0], size=n_samples, p=[0.95, 0.03, 0.02]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.80, 0.15, 0.05]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.70, 0.15, 0.15]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.80, 0.15, 0.05]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.80, 0.15, 0.05]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.90, 0.08, 0.02]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.85, 0.10, 0.05]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.70, 0.20, 0.10]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.25, 0.25, 0.50]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.80, 0.15, 0.05]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.70, 0.20, 0.10]),
            np.random.choice([2, 1, 0], size=n_samples, p=[0.80, 0.15, 0.05]),
        ],
        default=0
    )

    # Grunting depends on LungParench and Sick
    data["Grunting"] = np.select(
        [
            (data["LungParench"] == 2) & (data["Sick"] == 1),  # Normal, yes
            (data["LungParench"] == 1) & (data["Sick"] == 1),  # Congested, yes
            (data["LungParench"] == 0) & (data["Sick"] == 1),  # Abnormal, yes
            (data["LungParench"] == 2) & (data["Sick"] == 0),  # Normal, no
            (data["LungParench"] == 1) & (data["Sick"] == 0),  # Congested, no
            (data["LungParench"] == 0) & (data["Sick"] == 0),  # Abnormal, no
        ],
        [
            np.random.choice([1, 0], size=n_samples, p=[0.2, 0.8]),
            np.random.choice([1, 0], size=n_samples, p=[0.4, 0.6]),
            np.random.choice([1, 0], size=n_samples, p=[0.8, 0.2]),
            np.random.choice([1, 0], size=n_samples, p=[0.05, 0.95]),
            np.random.choice([1, 0], size=n_samples, p=[0.2, 0.8]),
            np.random.choice([1, 0], size=n_samples, p=[0.6, 0.4]),
        ],
        default=0
    )

    # GruntingReport depends on Grunting
    data["GruntingReport"] = np.where(data["Grunting"] == 1, 
                                    np.random.choice([1, 0], size=n_samples, p=[0.8, 0.2]), 
                                    np.random.choice([1, 0], size=n_samples, p=[0.1, 0.9]))

    # ChestXray depends on LungParench and LungFlow
    data["ChestXray"] = np.select(
        [
            (data["LungParench"] == 2) & (data["LungFlow"] == 2),  # Normal, Normal
            (data["LungParench"] == 1) & (data["LungFlow"] == 2),  # Congested, Normal
            (data["LungParench"] == 0) & (data["LungFlow"] == 2),  # Abnormal, Normal
            (data["LungParench"] == 2) & (data["LungFlow"] == 1),  # Normal, Low
            (data["LungParench"] == 1) & (data["LungFlow"] == 1),  # Congested, Low
            (data["LungParench"] == 0) & (data["LungFlow"] == 1),  # Abnormal, Low
            (data["LungParench"] == 2) & (data["LungFlow"] == 0),  # Normal, High
            (data["LungParench"] == 1) & (data["LungFlow"] == 0),  # Congested, High
            (data["LungParench"] == 0) & (data["LungFlow"] == 0),  # Abnormal, High
        ],
        [
            np.random.choice([4, 3, 2, 1, 0], size=n_samples, p=[0.90, 0.03, 0.03, 0.01, 0.03]),
            np.random.choice([4, 3, 2, 1, 0], size=n_samples, p=[0.05, 0.02, 0.15, 0.70, 0.08]),
            np.random.choice([4, 3, 2, 1, 0], size=n_samples, p=[0.05, 0.05, 0.05, 0.05, 0.80]),
            np.random.choice([4, 3, 2, 1, 0], size=n_samples, p=[0.14, 0.80, 0.02, 0.02, 0.02]),
            np.random.choice([4, 3, 2, 1, 0], size=n_samples, p=[0.05, 0.22, 0.08, 0.50, 0.15]),
            np.random.choice([4, 3, 2, 1, 0], size=n_samples, p=[0.05, 0.15, 0.05, 0.05, 0.70]),
            np.random.choice([4, 3, 2, 1, 0], size=n_samples, p=[0.15, 0.01, 0.79, 0.04, 0.01]),
            np.random.choice([4, 3, 2, 1, 0], size=n_samples, p=[0.05, 0.02, 0.40, 0.40, 0.13]),
            np.random.choice([4, 3, 2, 1, 0], size=n_samples, p=[0.24, 0.33, 0.03, 0.34, 0.06]),
        ],
        default=0
    )

    # XrayReport depends on ChestXray
    data["XrayReport"] = np.select(
        [
            data["ChestXray"] == 4,  # Normal
            data["ChestXray"] == 3,  # Oligaemic
            data["ChestXray"] == 2,  # Plethoric
            data["ChestXray"] == 1,  # Grd_Glass
            data["ChestXray"] == 0,  # Asy/Patch
        ],
        [
            np.random.choice([4, 3, 2, 1, 0], size=n_samples, p=[0.80, 0.06, 0.06, 0.02, 0.06]),
            np.random.choice([4, 3, 2, 1, 0], size=n_samples, p=[0.10, 0.80, 0.02, 0.02, 0.06]),
            np.random.choice([4, 3, 2, 1, 0], size=n_samples, p=[0.10, 0.02, 0.80, 0.02, 0.06]),
            np.random.choice([4, 3, 2, 1, 0], size=n_samples, p=[0.08, 0.02, 0.10, 0.60, 0.20]),
            np.random.choice([4, 3, 2, 1, 0], size=n_samples, p=[0.08, 0.02, 0.10, 0.10, 0.70]),
        ],
        default=0
    )

    end_time_child_config = time.time()

    print(f"Time to config dataset CHILD {i}: ", end_time_child_config - start_time_child_config)

    return data

V = [
    [1, 0], # yes, no
    [1, 0], # Equal, Unequal
    [2, 1, 0], # Mild, Moderate, Severe
    [2, 1, 0], # Normal, Low, High
    [4, 3, 2, 1, 0], # Normal, Oligaemic, Plethoric, Grd_Glass, Asy/Patch
    [1, 0], # yes, no
    [1, 0], # yes, no
    [2, 1, 0], # <5, 5-12, 12+
    [2, 1, 0], # <5, 5-12, 12+
    [1, 0], # <7.5, >=7.5
    [4, 3, 2, 1, 0], # Normal, Oligaemic, Plethoric, Grd_Glass, Asy/Patchy
    [5, 4, 3, 2, 1, 0], # PFC, TGA, Fallot, PAIVS, TAPVD, Lung
    [1, 0], # yes, no
    [2, 1, 0], # 0-3_days, 4-10_days, 11-30_days
    [1, 0], # yes, no
    [2, 1, 0], # Lt_to_Rt, None, Rt_to_Lt
    [3, 2, 1, 0], # None, Mild, Complete, Transp.
    [2, 1, 0], # Normal, Congested, Abnormal
    [2, 1, 0], # Normal, Low, High
    [1, 0] # yes, no
]

r = np.array([len(values) for values in V])

expected_node_configuration_child = [
    {'id': 0, 'name': 'BirthAsphyxia', 'parents': []}, 
    {'id': 1, 'name': 'Disease', 'parents': [0]}, 
    {'id': 2, 'name': 'Sick', 'parents': [1]}, 
    {'id': 3, 'name': 'DuctFlow', 'parents': [1]}, 
    {'id': 4, 'name': 'CardiacMixing', 'parents': [1]}, 
    {'id': 5, 'name': 'LungParench', 'parents': [1]},
    {'id': 6, 'name': 'LungFlow', 'parents': [1]}, 
    {'id': 7, 'name': 'LVH', 'parents': [1]}, 
    {'id': 8, 'name': 'Age', 'parents': [1, 2]}, 
    {'id': 9, 'name': 'Grunting', 'parents': [5, 2]}, 
    {'id': 10, 'name': 'HypDistrib', 'parents': [3, 4]}, 
    {'id': 11, 'name': 'HypoxiaInO2', 'parents': [4, 5]}, 
    {'id': 12, 'name': 'CO2', 'parents': [5]}, 
    {'id': 13, 'name': 'ChestXray', 'parents': [5, 6]}, 
    {'id': 14, 'name': 'LVHreport', 'parents': [7]}, 
    {'id': 15, 'name': 'GruntingReport', 'parents': [9]}, 
    {'id': 16, 'name': 'LowerBodyO2', 'parents': [10, 11]}, 
    {'id': 17, 'name': 'RUQO2', 'parents': [11]}, 
    {'id': 18, 'name': 'CO2Report', 'parents': [12]}, 
    {'id': 19, 'name': 'XrayReport', 'parents': [13]}
]

network_counts_child = {}
network_structure_difference_child = {}