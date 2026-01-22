```markdown
# Анализ данных новостей Lenta.ru

## Описание проекта
Проект включает в себя:
1. **Сбор данных** – парсинг новостей с сайта Lenta.ru по 4 категориям
2. **Классификация** – ML‑модель для определения категории новости по тексту
3. **Анализ** – исследование полученных данных

## Структура проекта
```text
project/
├── TASK_5.ipynb                  # Скрипт сбора данных
├── Task_6_kNN_Classifier.ipynb   # ML модель классификации
├── lenta_parser.py               # Скрипт парсера (если есть)
├── simple_lenta_news.csv         # Датасет с новостями
├── saved_models/                 # Сохраненные ML модели
│   ├── knn_lenta_classifier.pkl
│   └── tfidf_vectorizer.pkl
├── requirements.txt              # Зависимости
└── README.md                     # Этот файл
```

## Установка и запуск

### 1. Клонирование репозитория
```bash
git clone https://github.com/BorisITZaitsev/LentaRU_DataAnalysis.git
cd LentaRU_DataAnalysis
```

### 2. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 3. Запуск проекта

Откройте Jupyter Notebook:
```bash
jupyter notebook
```

Или запустите соответствующие скрипты напрямую через Python:
```bash
python lenta_parser.py
```

## Категории новостей
- **economics** (Экономика)
- **science** (Наука и техника)
- **culture** (Культура)
- **sport** (Спорт)

## Используемые технологии
- Python 3.x  
- Jupyter Notebook  
- Scikit-learn (ML)  
- Pandas, NumPy (анализ данных)  
- BeautifulSoup4 (парсинг)  
- Matplotlib, Seaborn (визуализация)

## Автор
[BorisITZaitsev]

## Лицензия
MIT License

---

## Создание файла requirements.txt

#### 4. Создайте `requirements.txt`

В терминале (например, в PyCharm) выполните:

```bash
# Активируйте виртуальную среду (если используете)

# Для Windows:
venv\Scripts\activate

# Для Mac/Linux:
source venv/bin/activate

# Создайте requirements.txt
pip freeze > requirements.txt
```
```