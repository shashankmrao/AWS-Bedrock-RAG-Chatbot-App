AWS CloudShell:
python3 --version #Should be same version as lambda function runtime
mkdir ~/lambda-layer
cd ~/lambda-layer

python3 -m venv buildenv
source buildenv/bin/activate

pip install --upgrade pip

mkdir python
#upload requirements.txt under Actions
mv ~/requirements.txt .
pip install -r requirements.txt -t python

find python -name "*pydantic_core*"
If you do not see _pydantic_core.cpython-313-x86_64-linux-gnu.so, then the layer will fail in Lambda.

PYTHONPATH=python python3
import pydantic_core
import pydantic
import langchain_core

print("SUCCESS")
exit()

zip -r layer.zip python
ls -lh layer.zip
#Download file with path lambda-layer/layer.zip under Actions

Create inline policy for the IAM role not having retrieve permission
in json format
{
  "Effect": "Allow",
  "Action": [
    "bedrock:Retrieve",
    "bedrock:RetrieveAndGenerate",
    "bedrock:InvokeModel"
  ],
  "Resource": "*"
}

