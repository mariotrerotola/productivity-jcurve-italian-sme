from pathlib import Path

# Path Configurations
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'dati'
OUTPUT_DIR = BASE_DIR / 'figure'
RESULTS_DIR = BASE_DIR / 'risultati'

# Data Files
DATA_FILE = DATA_DIR / 'Aida_Export_2.xls'
COL_ATECO = 78 # ATECO 2007 Column (0-based index)

# ATECO Sector Map (2 digits)
SECTOR_MAP = {
    '26': 'Computer & Electronics',
    '27': 'Electrical Equipment', 
    '28': 'Machinery'
}

# Province -> Macro Areas Map
PROVINCE_TO_MACRO = {
    'MO': 'Nord', 'MI': 'Nord', 'BG': 'Nord', 'BS': 'Nord', 'PD': 'Nord', 
    'VR': 'Nord', 'TO': 'Nord', 'CN': 'Nord', 'VI': 'Nord', 'TV': 'Nord',
    'UD': 'Nord', 'TN': 'Nord', 'BZ': 'Nord', 'VA': 'Nord', 'CO': 'Nord',
    'MB': 'Nord', 'LC': 'Nord', 'LO': 'Nord', 'CR': 'Nord', 'PV': 'Nord',
    'MN': 'Nord', 'SO': 'Nord', 'NO': 'Nord', 'BI': 'Nord', 'VC': 'Nord',
    'AL': 'Nord', 'AT': 'Nord', 'AO': 'Nord', 'GE': 'Nord', 'SV': 'Nord',
    'IM': 'Nord', 'SP': 'Nord', 'PC': 'Nord', 'PR': 'Nord', 'RE': 'Nord',
    'BO': 'Nord', 'FE': 'Nord', 'RA': 'Nord', 'FC': 'Nord', 'RN': 'Nord',
    'TS': 'Nord', 'GO': 'Nord', 'PN': 'Nord', 'BL': 'Nord', 'VB': 'Nord',
    
    # Non-standard AIDA codes (Verified via company name inspection)
    'VE': 'Nord',    # Venezia/Veneto (Hinowa, Areva)
    'BE': 'Nord',    # Bergamo (Italian Cable Co, OMEFA)
    'CU': 'Nord',    # Cuneo (Manitowoc, Nord Engineering)
    'MA': 'Nord',    # Mantova (Lavorwash, Pecso) -> NOT Matera
    'FO': 'Nord',    # Forlì (Emicon, Valli)
    
    'FI': 'Centro', 'PO': 'Centro', 'PT': 'Centro', 'LU': 'Centro', 'MS': 'Centro',
    'PI': 'Centro', 'LI': 'Centro', 'GR': 'Centro', 'SI': 'Centro', 'AR': 'Centro',
    'PG': 'Centro', 'TR': 'Centro', 'AN': 'Centro', 'PU': 'Centro', 'MC': 'Centro',
    'FM': 'Centro', 'AP': 'Centro', 'RM': 'Centro', 'VT': 'Centro', 'RI': 'Centro',
    'LT': 'Centro', 'FR': 'Centro', 'TR': 'Centro', 'PU': 'Centro',
    
    # Verified Center AIDA codes
    'RO': 'Centro',  # ROMA (Northrop, Larimart, Convert) -> NOT Rovigo
    'AS': 'Centro',  # Ascoli Piceno (Inim, Unionalpha) -> NOT Asti
    'LA': 'Centro',  # Latina (Music & Lights, Adicomp)
    
    'NA': 'Sud', 'AV': 'Sud', 'BN': 'Sud', 'CE': 'Sud', 'SA': 'Sud',
    'BA': 'Sud', 'FG': 'Sud', 'BR': 'Sud', 'LE': 'Sud', 'TA': 'Sud', 'BT': 'Sud',
    'PZ': 'Sud', 'MT': 'Sud', 'CB': 'Sud', 'IS': 'Sud', 'AQ': 'Sud',
    'TE': 'Sud', 'PE': 'Sud', 'CH': 'Sud', 'CZ': 'Sud', 'CS': 'Sud',
    'KR': 'Sud', 'RC': 'Sud', 'VV': 'Sud', 'PA': 'Sud', 'CT': 'Sud',
    'ME': 'Sud', 'AG': 'Sud', 'CL': 'Sud', 'EN': 'Sud', 'RG': 'Sud',
    'SR': 'Sud', 'TP': 'Sud', 'CA': 'Sud', 'NU': 'Sud', 'OR': 'Sud',
    'SS': 'Sud', 'SU': 'Sud', 
    "L'": 'Sud',     # L'Aquila (Saes, Elco)
    
    'FI': 'Centro', 'PO': 'Centro', 'PT': 'Centro', 'LU': 'Centro', 'MS': 'Centro',
    'PI': 'Centro', 'LI': 'Centro', 'GR': 'Centro', 'SI': 'Centro', 'AR': 'Centro',
    'PG': 'Centro', 'TR': 'Centro', 'AN': 'Centro', 'PU': 'Centro', 'MC': 'Centro',
    'FM': 'Centro', 'AP': 'Centro', 'RM': 'Centro', 'VT': 'Centro', 'RI': 'Centro',
    'LT': 'Centro', 'FR': 'Centro',
    
    'NA': 'Sud', 'AV': 'Sud', 'BN': 'Sud', 'CE': 'Sud', 'SA': 'Sud',
    'BA': 'Sud', 'FG': 'Sud', 'BR': 'Sud', 'LE': 'Sud', 'TA': 'Sud', 'BT': 'Sud',
    'PZ': 'Sud', 'MT': 'Sud', 'CB': 'Sud', 'IS': 'Sud', 'AQ': 'Sud',
    'TE': 'Sud', 'PE': 'Sud', 'CH': 'Sud', 'CZ': 'Sud', 'CS': 'Sud',
    'KR': 'Sud', 'RC': 'Sud', 'VV': 'Sud', 'PA': 'Sud', 'CT': 'Sud',
    'ME': 'Sud', 'AG': 'Sud', 'CL': 'Sud', 'EN': 'Sud', 'RG': 'Sud',
    'SR': 'Sud', 'TP': 'Sud', 'CA': 'Sud', 'NU': 'Sud', 'OR': 'Sud',
    'SS': 'Sud', 'SU': 'Sud'
}

# Graphic Configurations
PLOT_STYLE = 'seaborn-v0_8-whitegrid'
ACADEMIC_COLORS = ['#2c2c2c', '#6b6b6b', '#a8a8a8']
