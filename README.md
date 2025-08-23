
## AI Tutor

## Prerequisites

- Python 3.11 or later
- [pip](https://pip.pypa.io/en/stable/)
- (Optional) [venv](https://docs.python.org/3/library/venv.html) for creating virtual environments

## Setup Instructions

1. **Clone the repository (if needed)**
   ```bash
   git clone <repository-url>
   cd DeepSeek_chatbot
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   ```
   On Windows, activate the virtual environment with:
   ```bash
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Create a `.env` file in the root folder (if not already present) and add the following variables. Replace the placeholder values with your actual Azure endpoint and key:
   ```env
   AZURE_ENDPOINT=https://your-azure-endpoint
   AZURE_KEY=your-azure-key
   ```

## Running the App

The repository contains two main Python files:

- **`chat.py`**: A standalone script demonstrating a simple Azure chat call.
- **`app.py`**: A Flask web application that streams chat responses to a web client.

To run the Flask web app locally, execute:
```bash
python app.py
```
By default, the app will start on `http://127.0.0.1:5000/`.

## Using the App

- Open your browser and navigate to `http://127.0.0.1:5000/`.
- Type your message in the input box and press **Send**.
- The app will display the response from Azure along with internal reasoning wrapped in `<think>...</think>`.

## Troubleshooting

- **Environment Variables:** Ensure your `.env` file is in the same directory as `app.py` and contains correct values.
- **Dependency Issues:** If installation of dependencies fails, ensure you are using the correct Python version.


