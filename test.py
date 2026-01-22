import sys
print(f"Python path: {sys.executable}")
print(f"Version: {sys.version}")

import requests
import pandas as pd
from bs4 import BeautifulSoup

print("\nВсе библиотеки загружены успешно!")
print(f"Requests version: {requests.__version__}")
print(f"Pandas version: {pd.__version__}")