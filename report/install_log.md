# Python 3.13 패키지 설치 로그

실행 시간: 2025-05-19 06:12:34

## 1. 호환 패키지 설치

```
$ C:\Python313\python.exe -m pip install boto3
Defaulting to user installation because normal site-packages is not writeable
Collecting boto3
  Using cached boto3-1.38.18-py3-none-any.whl.metadata (6.6 kB)
Collecting botocore<1.39.0,>=1.38.18 (from boto3)
  Downloading botocore-1.38.18-py3-none-any.whl.metadata (5.7 kB)
Collecting jmespath<2.0.0,>=0.7.1 (from boto3)
  Downloading jmespath-1.0.1-py3-none-any.whl.metadata (7.6 kB)
Collecting s3transfer<0.13.0,>=0.12.0 (from boto3)
  Downloading s3transfer-0.12.0-py3-none-any.whl.metadata (1.7 kB)
Collecting python-dateutil<3.0.0,>=2.1 (from botocore<1.39.0,>=1.38.18->boto3)
  Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Requirement already satisfied: urllib3!=2.2.0,<3,>=1.25.4 in c:\users\packr\appdata\roaming\python\python313\site-packages (from botocore<1.39.0,>=1.38.18->boto3) (2.4.0)
Collecting six>=1.5 (from python-dateutil<3.0.0,>=2.1->botocore<1.39.0,>=1.38.18->boto3)
  Using cached six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Downloading boto3-1.38.18-py3-none-any.whl (139 kB)
Downloading botocore-1.38.18-py3-none-any.whl (13.6 MB)
   ---------------------------------------- 13.6/13.6 MB 33.7 MB/s eta 0:00:00
Downloading jmespath-1.0.1-py3-none-any.whl (20 kB)
Downloading s3transfer-0.12.0-py3-none-any.whl (84 kB)
Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Using cached six-1.17.0-py2.py3-none-any.whl (11 kB)
Installing collected packages: six, jmespath, python-dateutil, botocore, s3transfer, boto3
Successfully installed boto3-1.38.18 botocore-1.38.18 jmespath-1.0.1 python-dateutil-2.9.0.post0 s3transfer-0.12.0 six-1.17.0

$ C:\Python313\python.exe -m pip install codex-cli
Defaulting to user installation because normal site-packages is not writeable
Collecting codex-cli
  Using cached codex_cli-1.0.1-py3-none-any.whl.metadata (1.1 kB)
Collecting google-generativeai (from codex-cli)
  Downloading google_generativeai-0.8.5-py3-none-any.whl.metadata (3.9 kB)
Collecting Pillow (from codex-cli)
  Downloading pillow-11.2.1-cp313-cp313-win_amd64.whl.metadata (9.1 kB)
Collecting python-dotenv (from codex-cli)
  Using cached python_dotenv-1.1.0-py3-none-any.whl.metadata (24 kB)
Collecting google-ai-generativelanguage==0.6.15 (from google-generativeai->codex-cli)
  Downloading google_ai_generativelanguage-0.6.15-py3-none-any.whl.metadata (5.7 kB)
Collecting google-api-core (from google-generativeai->codex-cli)
  Downloading google_api_core-2.24.2-py3-none-any.whl.metadata (3.0 kB)
Collecting google-api-python-client (from google-generativeai->codex-cli)
  Downloading google_api_python_client-2.169.0-py3-none-any.whl.metadata (6.7 kB)
Collecting google-auth>=2.15.0 (from google-generativeai->codex-cli)
  Using cached google_auth-2.40.1-py2.py3-none-any.whl.metadata (6.2 kB)
Collecting protobuf (from google-generativeai->codex-cli)
  Using cached protobuf-6.31.0-cp310-abi3-win_amd64.whl.metadata (593 bytes)
Collecting pydantic (from google-generativeai->codex-cli)
  Using cached pydantic-2.11.4-py3-none-any.whl.metadata (66 kB)
Collecting tqdm (from google-generativeai->codex-cli)
  Using cached tqdm-4.67.1-py3-none-any.whl.metadata (57 kB)
Collecting typing-extensions (from google-generativeai->codex-cli)
  Using cached typing_extensions-4.13.2-py3-none-any.whl.metadata (3.0 kB)
Collecting google-api-core (from google-generativeai->codex-cli)
  Downloading google_api_core-2.25.0rc1-py3-none-any.whl.metadata (3.0 kB)
Collecting proto-plus<2.0.0dev,>=1.22.3 (from google-ai-generativelanguage==0.6.15->google-generativeai->codex-cli)
  Downloading proto_plus-1.26.1-py3-none-any.whl.metadata (2.2 kB)
Collecting protobuf (from google-generativeai->codex-cli)
  Using cached protobuf-5.29.4-cp310-abi3-win_amd64.whl.metadata (592 bytes)
Collecting googleapis-common-protos<2.0.0,>=1.56.2 (from google-api-core->google-generativeai->codex-cli)
  Using cached googleapis_common_protos-1.70.0-py3-none-any.whl.metadata (9.3 kB)
Requirement already satisfied: requests<3.0.0,>=2.18.0 in c:\users\packr\appdata\roaming\python\python313\site-packages (from google-api-core->google-generativeai->codex-cli) (2.32.3)
Collecting cachetools<6.0,>=2.0.0 (from google-auth>=2.15.0->google-generativeai->codex-cli)
  Using cached cachetools-5.5.2-py3-none-any.whl.metadata (5.4 kB)
Collecting pyasn1-modules>=0.2.1 (from google-auth>=2.15.0->google-generativeai->codex-cli)
  Using cached pyasn1_modules-0.4.2-py3-none-any.whl.metadata (3.5 kB)
Collecting rsa<5,>=3.1.4 (from google-auth>=2.15.0->google-generativeai->codex-cli)
  Using cached rsa-4.9.1-py3-none-any.whl.metadata (5.6 kB)
Collecting httplib2<1.0.0,>=0.19.0 (from google-api-python-client->google-generativeai->codex-cli)
  Downloading httplib2-0.22.0-py3-none-any.whl.metadata (2.6 kB)
Collecting google-auth-httplib2<1.0.0,>=0.2.0 (from google-api-python-client->google-generativeai->codex-cli)
  Downloading google_auth_httplib2-0.2.0-py2.py3-none-any.whl.metadata (2.2 kB)
Collecting uritemplate<5,>=3.0.1 (from google-api-python-client->google-generativeai->codex-cli)
  Downloading uritemplate-4.1.1-py2.py3-none-any.whl.metadata (2.9 kB)
Collecting annotated-types>=0.6.0 (from pydantic->google-generativeai->codex-cli)
  Using cached annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
Collecting pydantic-core==2.33.2 (from pydantic->google-generativeai->codex-cli)
  Using cached pydantic_core-2.33.2-cp313-cp313-win_amd64.whl.metadata (6.9 kB)
Collecting typing-inspection>=0.4.0 (from pydantic->google-generativeai->codex-cli)
  Using cached typing_inspection-0.4.0-py3-none-any.whl.metadata (2.6 kB)
Collecting colorama (from tqdm->google-generativeai->codex-cli)
  Using cached colorama-0.4.6-py2.py3-none-any.whl.metadata (17 kB)
Collecting grpcio<2.0.0,>=1.33.2 (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-ai-generativelanguage==0.6.15->google-generativeai->codex-cli)
  Using cached grpcio-1.71.0-cp313-cp313-win_amd64.whl.metadata (4.0 kB)
Collecting grpcio-status<2.0.0,>=1.33.2 (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-ai-generativelanguage==0.6.15->google-generativeai->codex-cli)
  Downloading grpcio_status-1.71.0-py3-none-any.whl.metadata (1.1 kB)
Collecting pyparsing!=3.0.0,!=3.0.1,!=3.0.2,!=3.0.3,<4,>=2.4.2 (from httplib2<1.0.0,>=0.19.0->google-api-python-client->google-generativeai->codex-cli)
  Using cached pyparsing-3.2.3-py3-none-any.whl.metadata (5.0 kB)
Collecting pyasn1<0.7.0,>=0.6.1 (from pyasn1-modules>=0.2.1->google-auth>=2.15.0->google-generativeai->codex-cli)
  Using cached pyasn1-0.6.1-py3-none-any.whl.metadata (8.4 kB)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\packr\appdata\roaming\python\python313\site-packages (from requests<3.0.0,>=2.18.0->google-api-core->google-generativeai->codex-cli) (3.4.2)
Requirement already satisfied: idna<4,>=2.5 in c:\users\packr\appdata\roaming\python\python313\site-packages (from requests<3.0.0,>=2.18.0->google-api-core->google-generativeai->codex-cli) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\packr\appdata\roaming\python\python313\site-packages (from requests<3.0.0,>=2.18.0->google-api-core->google-generativeai->codex-cli) (2.4.0)
Requirement already satisfied: certifi>=2017.4.17 in c:\users\packr\appdata\roaming\python\python313\site-packages (from requests<3.0.0,>=2.18.0->google-api-core->google-generativeai->codex-cli) (2025.4.26)
Downloading codex_cli-1.0.1-py3-none-any.whl (2.6 kB)
Downloading google_generativeai-0.8.5-py3-none-any.whl (155 kB)
Downloading google_ai_generativelanguage-0.6.15-py3-none-any.whl (1.3 MB)
   ---------------------------------------- 1.3/1.3 MB 7.1 MB/s eta 0:00:00
Downloading pillow-11.2.1-cp313-cp313-win_amd64.whl (2.7 MB)
   ---------------------------------------- 2.7/2.7 MB 22.9 MB/s eta 0:00:00
Using cached python_dotenv-1.1.0-py3-none-any.whl (20 kB)
Downloading google_api_core-2.25.0rc1-py3-none-any.whl (160 kB)
Using cached google_auth-2.40.1-py2.py3-none-any.whl (216 kB)
Using cached protobuf-5.29.4-cp310-abi3-win_amd64.whl (434 kB)
Downloading google_api_python_client-2.169.0-py3-none-any.whl (13.3 MB)
   ---------------------------------------- 13.3/13.3 MB 34.2 MB/s eta 0:00:00
Using cached pydantic-2.11.4-py3-none-any.whl (443 kB)
Using cached pydantic_core-2.33.2-cp313-cp313-win_amd64.whl (2.0 MB)
Using cached typing_extensions-4.13.2-py3-none-any.whl (45 kB)
Using cached tqdm-4.67.1-py3-none-any.whl (78 kB)
Using cached annotated_types-0.7.0-py3-none-any.whl (13 kB)
Using cached cachetools-5.5.2-py3-none-any.whl (10 kB)
Downloading google_auth_httplib2-0.2.0-py2.py3-none-any.whl (9.3 kB)
Using cached googleapis_common_protos-1.70.0-py3-none-any.whl (294 kB)
Downloading httplib2-0.22.0-py3-none-any.whl (96 kB)
Downloading proto_plus-1.26.1-py3-none-any.whl (50 kB)
Using cached pyasn1_modules-0.4.2-py3-none-any.whl (181 kB)
Using cached rsa-4.9.1-py3-none-any.whl (34 kB)
Using cached typing_inspection-0.4.0-py3-none-any.whl (14 kB)
Downloading uritemplate-4.1.1-py2.py3-none-any.whl (10 kB)
Using cached colorama-0.4.6-py2.py3-none-any.whl (25 kB)
Using cached grpcio-1.71.0-cp313-cp313-win_amd64.whl (4.3 MB)
Downloading grpcio_status-1.71.0-py3-none-any.whl (14 kB)
Using cached pyasn1-0.6.1-py3-none-any.whl (83 kB)
Using cached pyparsing-3.2.3-py3-none-any.whl (111 kB)
Installing collected packages: uritemplate, typing-extensions, python-dotenv, pyparsing, pyasn1, protobuf, Pillow, grpcio, colorama, cachetools, annotated-types, typing-inspection, tqdm, rsa, pydantic-core, pyasn1-modules, proto-plus, httplib2, googleapis-common-protos, pydantic, grpcio-status, google-auth, google-auth-httplib2, google-api-core, google-api-python-client, google-ai-generativelanguage, google-generativeai, codex-cli
Successfully installed Pillow-11.2.1 annotated-types-0.7.0 cachetools-5.5.2 codex-cli-1.0.1 colorama-0.4.6 google-ai-generativelanguage-0.6.15 google-api-core-2.25.0rc1 google-api-python-client-2.169.0 google-auth-2.40.1 google-auth-httplib2-0.2.0 google-generativeai-0.8.5 googleapis-common-protos-1.70.0 grpcio-1.71.0 grpcio-status-1.71.0 httplib2-0.22.0 proto-plus-1.26.1 protobuf-5.29.4 pyasn1-0.6.1 pyasn1-modules-0.4.2 pydantic-2.11.4 pydantic-core-2.33.2 pyparsing-3.2.3 python-dotenv-1.1.0 rsa-4.9.1 tqdm-4.67.1 typing-extensions-4.13.2 typing-inspection-0.4.0 uritemplate-4.1.1

$ C:\Python313\python.exe -m pip install dash
Defaulting to user installation because normal site-packages is not writeable
Collecting dash
  Using cached dash-3.0.4-py3-none-any.whl.metadata (10 kB)
Collecting Flask<3.1,>=1.0.4 (from dash)
  Using cached flask-3.0.3-py3-none-any.whl.metadata (3.2 kB)
Collecting Werkzeug<3.1 (from dash)
  Using cached werkzeug-3.0.6-py3-none-any.whl.metadata (3.7 kB)
Collecting plotly>=5.0.0 (from dash)
  Using cached plotly-6.1.0-py3-none-any.whl.metadata (6.9 kB)
Collecting importlib-metadata (from dash)
  Using cached importlib_metadata-8.7.0-py3-none-any.whl.metadata (4.8 kB)
Requirement already satisfied: typing-extensions>=4.1.1 in c:\users\packr\appdata\roaming\python\python313\site-packages (from dash) (4.13.2)
Requirement already satisfied: requests in c:\users\packr\appdata\roaming\python\python313\site-packages (from dash) (2.32.3)
Collecting retrying (from dash)
  Using cached retrying-1.3.4-py3-none-any.whl.metadata (6.9 kB)
Collecting nest-asyncio (from dash)
  Using cached nest_asyncio-1.6.0-py3-none-any.whl.metadata (2.8 kB)
Collecting setuptools (from dash)
  Using cached setuptools-80.7.1-py3-none-any.whl.metadata (6.6 kB)
Collecting Jinja2>=3.1.2 (from Flask<3.1,>=1.0.4->dash)
  Using cached jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
Collecting itsdangerous>=2.1.2 (from Flask<3.1,>=1.0.4->dash)
  Using cached itsdangerous-2.2.0-py3-none-any.whl.metadata (1.9 kB)
Collecting click>=8.1.3 (from Flask<3.1,>=1.0.4->dash)
  Using cached click-8.2.0-py3-none-any.whl.metadata (2.5 kB)
Collecting blinker>=1.6.2 (from Flask<3.1,>=1.0.4->dash)
  Using cached blinker-1.9.0-py3-none-any.whl.metadata (1.6 kB)
Collecting narwhals>=1.15.1 (from plotly>=5.0.0->dash)
  Using cached narwhals-1.39.1-py3-none-any.whl.metadata (11 kB)
Requirement already satisfied: packaging in c:\users\packr\appdata\roaming\python\python313\site-packages (from plotly>=5.0.0->dash) (25.0)
Collecting MarkupSafe>=2.1.1 (from Werkzeug<3.1->dash)
  Using cached MarkupSafe-3.0.2-cp313-cp313-win_amd64.whl.metadata (4.1 kB)
Collecting zipp>=3.20 (from importlib-metadata->dash)
  Using cached zipp-3.21.0-py3-none-any.whl.metadata (3.7 kB)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\packr\appdata\roaming\python\python313\site-packages (from requests->dash) (3.4.2)
Requirement already satisfied: idna<4,>=2.5 in c:\users\packr\appdata\roaming\python\python313\site-packages (from requests->dash) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\packr\appdata\roaming\python\python313\site-packages (from requests->dash) (2.4.0)
Requirement already satisfied: certifi>=2017.4.17 in c:\users\packr\appdata\roaming\python\python313\site-packages (from requests->dash) (2025.4.26)
Requirement already satisfied: six>=1.7.0 in c:\users\packr\appdata\roaming\python\python313\site-packages (from retrying->dash) (1.17.0)
Requirement already satisfied: colorama in c:\users\packr\appdata\roaming\python\python313\site-packages (from click>=8.1.3->Flask<3.1,>=1.0.4->dash) (0.4.6)
Using cached dash-3.0.4-py3-none-any.whl (7.9 MB)
Using cached flask-3.0.3-py3-none-any.whl (101 kB)
Using cached plotly-6.1.0-py3-none-any.whl (16.1 MB)
Using cached werkzeug-3.0.6-py3-none-any.whl (227 kB)
Downloading importlib_metadata-8.7.0-py3-none-any.whl (27 kB)
Using cached nest_asyncio-1.6.0-py3-none-any.whl (5.2 kB)
Using cached retrying-1.3.4-py3-none-any.whl (11 kB)
Using cached setuptools-80.7.1-py3-none-any.whl (1.2 MB)
Using cached blinker-1.9.0-py3-none-any.whl (8.5 kB)
Downloading click-8.2.0-py3-none-any.whl (102 kB)
Using cached itsdangerous-2.2.0-py3-none-any.whl (16 kB)
Using cached jinja2-3.1.6-py3-none-any.whl (134 kB)
Using cached MarkupSafe-3.0.2-cp313-cp313-win_amd64.whl (15 kB)
Using cached narwhals-1.39.1-py3-none-any.whl (355 kB)
Using cached zipp-3.21.0-py3-none-any.whl (9.6 kB)
Installing collected packages: zipp, setuptools, retrying, nest-asyncio, narwhals, MarkupSafe, itsdangerous, click, blinker, Werkzeug, plotly, Jinja2, importlib-metadata, Flask, dash
Successfully installed Flask-3.0.3 Jinja2-3.1.6 MarkupSafe-3.0.2 Werkzeug-3.0.6 blinker-1.9.0 click-8.2.0 dash-3.0.4 importlib-metadata-8.7.0 itsdangerous-2.2.0 narwhals-1.39.1 nest-asyncio-1.6.0 plotly-6.1.0 retrying-1.3.4 setuptools-80.7.1 zipp-3.21.0

$ C:\Python313\python.exe -m pip install dash-cytoscape
Defaulting to user installation because normal site-packages is not writeable
Collecting dash-cytoscape
  Using cached dash_cytoscape-1.0.2.tar.gz (4.0 MB)
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: started
  Getting requirements to build wheel: finished with status 'done'
  Preparing metadata (pyproject.toml): started
  Preparing metadata (pyproject.toml): finished with status 'done'
Requirement already satisfied: dash in c:\users\packr\appdata\roaming\python\python313\site-packages (from dash-cytoscape) (3.0.4)
Requirement already satisfied: Flask<3.1,>=1.0.4 in c:\users\packr\appdata\roaming\python\python313\site-packages (from dash->dash-cytoscape) (3.0.3)
Requirement already satisfied: Werkzeug<3.1 in c:\users\packr\appdata\roaming\python\python313\site-packages (from dash->dash-cytoscape) (3.0.6)
Requirement already satisfied: plotly>=5.0.0 in c:\users\packr\appdata\roaming\python\python313\site-packages (from dash->dash-cytoscape) (6.1.0)
Requirement already satisfied: importlib-metadata in c:\users\packr\appdata\roaming\python\python313\site-packages (from dash->dash-cytoscape) (8.7.0)
Requirement already satisfied: typing-extensions>=4.1.1 in c:\users\packr\appdata\roaming\python\python313\site-packages (from dash->dash-cytoscape) (4.13.2)
Requirement already satisfied: requests in c:\users\packr\appdata\roaming\python\python313\site-packages (from dash->dash-cytoscape) (2.32.3)
Requirement already satisfied: retrying in c:\users\packr\appdata\roaming\python\python313\site-packages (from dash->dash-cytoscape) (1.3.4)
Requirement already satisfied: nest-asyncio in c:\users\packr\appdata\roaming\python\python313\site-packages (from dash->dash-cytoscape) (1.6.0)
Requirement already satisfied: setuptools in c:\users\packr\appdata\roaming\python\python313\site-packages (from dash->dash-cytoscape) (80.7.1)
Requirement already satisfied: Jinja2>=3.1.2 in c:\users\packr\appdata\roaming\python\python313\site-packages (from Flask<3.1,>=1.0.4->dash->dash-cytoscape) (3.1.6)
Requirement already satisfied: itsdangerous>=2.1.2 in c:\users\packr\appdata\roaming\python\python313\site-packages (from Flask<3.1,>=1.0.4->dash->dash-cytoscape) (2.2.0)
Requirement already satisfied: click>=8.1.3 in c:\users\packr\appdata\roaming\python\python313\site-packages (from Flask<3.1,>=1.0.4->dash->dash-cytoscape) (8.2.0)
Requirement already satisfied: blinker>=1.6.2 in c:\users\packr\appdata\roaming\python\python313\site-packages (from Flask<3.1,>=1.0.4->dash->dash-cytoscape) (1.9.0)
Requirement already satisfied: narwhals>=1.15.1 in c:\users\packr\appdata\roaming\python\python313\site-packages (from plotly>=5.0.0->dash->dash-cytoscape) (1.39.1)
Requirement already satisfied: packaging in c:\users\packr\appdata\roaming\python\python313\site-packages (from plotly>=5.0.0->dash->dash-cytoscape) (25.0)
Requirement already satisfied: MarkupSafe>=2.1.1 in c:\users\packr\appdata\roaming\python\python313\site-packages (from Werkzeug<3.1->dash->dash-cytoscape) (3.0.2)
Requirement already satisfied: zipp>=3.20 in c:\users\packr\appdata\roaming\python\python313\site-packages (from importlib-metadata->dash->dash-cytoscape) (3.21.0)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\packr\appdata\roaming\python\python313\site-packages (from requests->dash->dash-cytoscape) (3.4.2)
Requirement already satisfied: idna<4,>=2.5 in c:\users\packr\appdata\roaming\python\python313\site-packages (from requests->dash->dash-cytoscape) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\packr\appdata\roaming\python\python313\site-packages (from requests->dash->dash-cytoscape) (2.4.0)
Requirement already satisfied: certifi>=2017.4.17 in c:\users\packr\appdata\roaming\python\python313\site-packages (from requests->dash->dash-cytoscape) (2025.4.26)
Requirement already satisfied: six>=1.7.0 in c:\users\packr\appdata\roaming\python\python313\site-packages (from retrying->dash->dash-cytoscape) (1.17.0)
Requirement already satisfied: colorama in c:\users\packr\appdata\roaming\python\python313\site-packages (from click>=8.1.3->Flask<3.1,>=1.0.4->dash->dash-cytoscape) (0.4.6)
Building wheels for collected packages: dash-cytoscape
  Building wheel for dash-cytoscape (pyproject.toml): started
  Building wheel for dash-cytoscape (pyproject.toml): finished with status 'done'
  Created wheel for dash-cytoscape: filename=dash_cytoscape-1.0.2-py3-none-any.whl size=4010844 sha256=895993a35c4d24e8bd18d060116a84e91bc25b3559738963c708b4c31dc3efa9
  Stored in directory: c:\users\packr\appdata\local\pip\cache\wheels\1d\3a\14\e8bdf175039d5a4597973c9c04e7ba2d4d77420c62a1fc7c0d
Successfully built dash-cytoscape
Installing collected packages: dash-cytoscape
Successfully installed dash-cytoscape-1.0.2

$ C:\Python313\python.exe -m pip install neo4j
Defaulting to user installation because normal site-packages is not writeable
Collecting neo4j
  Using cached neo4j-5.28.1-py3-none-any.whl.metadata (5.9 kB)
Collecting pytz (from neo4j)
  Using cached pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
Using cached neo4j-5.28.1-py3-none-any.whl (312 kB)
Using cached pytz-2025.2-py2.py3-none-any.whl (509 kB)
Installing collected packages: pytz, neo4j
Successfully installed neo4j-5.28.1 pytz-2025.2

$ C:\Python313\python.exe -m pip install pandas
Defaulting to user installation because normal site-packages is not writeable
Collecting pandas
  Using cached pandas-2.2.3-cp313-cp313-win_amd64.whl.metadata (19 kB)
Collecting numpy>=1.26.0 (from pandas)
  Downloading numpy-2.2.6-cp313-cp313-win_amd64.whl.metadata (60 kB)
Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\packr\appdata\roaming\python\python313\site-packages (from pandas) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in c:\users\packr\appdata\roaming\python\python313\site-packages (from pandas) (2025.2)
Collecting tzdata>=2022.7 (from pandas)
  Using cached tzdata-2025.2-py2.py3-none-any.whl.metadata (1.4 kB)
Requirement already satisfied: six>=1.5 in c:\users\packr\appdata\roaming\python\python313\site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
Using cached pandas-2.2.3-cp313-cp313-win_amd64.whl (11.5 MB)
Downloading numpy-2.2.6-cp313-cp313-win_amd64.whl (12.6 MB)
   ---------------------------------------- 12.6/12.6 MB 33.1 MB/s eta 0:00:00
Using cached tzdata-2025.2-py2.py3-none-any.whl (347 kB)
Installing collected packages: tzdata, numpy, pandas
Successfully installed numpy-2.2.6 pandas-2.2.3 tzdata-2025.2

$ C:\Python313\python.exe -m pip install pathlib
Defaulting to user installation because normal site-packages is not writeable
Collecting pathlib
  Using cached pathlib-1.0.1-py3-none-any.whl.metadata (5.1 kB)
Downloading pathlib-1.0.1-py3-none-any.whl (14 kB)
Installing collected packages: pathlib
Successfully installed pathlib-1.0.1

$ C:\Python313\python.exe -m pip install plotly
Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: plotly in c:\users\packr\appdata\roaming\python\python313\site-packages (6.1.0)
Requirement already satisfied: narwhals>=1.15.1 in c:\users\packr\appdata\roaming\python\python313\site-packages (from plotly) (1.39.1)
Requirement already satisfied: packaging in c:\users\packr\appdata\roaming\python\python313\site-packages (from plotly) (25.0)

$ C:\Python313\python.exe -m pip install psycopg2-binary
Defaulting to user installation because normal site-packages is not writeable
Collecting psycopg2-binary
  Using cached psycopg2_binary-2.9.10-cp313-cp313-win_amd64.whl.metadata (4.8 kB)
Downloading psycopg2_binary-2.9.10-cp313-cp313-win_amd64.whl (2.6 MB)
   ---------------------------------------- 2.6/2.6 MB 27.1 MB/s eta 0:00:00
Installing collected packages: psycopg2-binary
Successfully installed psycopg2-binary-2.9.10

$ C:\Python313\python.exe -m pip install pyspark
Defaulting to user installation because normal site-packages is not writeable
Collecting pyspark
  Using cached pyspark-3.5.5.tar.gz (317.2 MB)
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: started
  Getting requirements to build wheel: finished with status 'done'
  Preparing metadata (pyproject.toml): started
  Preparing metadata (pyproject.toml): finished with status 'done'
Collecting py4j==0.10.9.7 (from pyspark)
  Downloading py4j-0.10.9.7-py2.py3-none-any.whl.metadata (1.5 kB)
Downloading py4j-0.10.9.7-py2.py3-none-any.whl (200 kB)
Building wheels for collected packages: pyspark
  Building wheel for pyspark (pyproject.toml): started
  Building wheel for pyspark (pyproject.toml): finished with status 'done'
  Created wheel for pyspark: filename=pyspark-3.5.5-py2.py3-none-any.whl size=317747966 sha256=614f0bb72641e709e8f037231a47f1b99335be2ae9f852a44ef7abcfa3e14761
  Stored in directory: c:\users\packr\appdata\local\pip\cache\wheels\9a\81\ff\d0378800053965023f8bdd676d306e93104e948cea3d1d5e70
Successfully built pyspark
Installing collected packages: py4j, pyspark
Successfully installed py4j-0.10.9.7 pyspark-3.5.5

$ C:\Python313\python.exe -m pip install pytest
Defaulting to user installation because normal site-packages is not writeable
Collecting pytest
  Using cached pytest-8.3.5-py3-none-any.whl.metadata (7.6 kB)
Requirement already satisfied: colorama in c:\users\packr\appdata\roaming\python\python313\site-packages (from pytest) (0.4.6)
Collecting iniconfig (from pytest)
  Downloading iniconfig-2.1.0-py3-none-any.whl.metadata (2.7 kB)
Requirement already satisfied: packaging in c:\users\packr\appdata\roaming\python\python313\site-packages (from pytest) (25.0)
Collecting pluggy<2,>=1.5 (from pytest)
  Downloading pluggy-1.6.0-py3-none-any.whl.metadata (4.8 kB)
Downloading pytest-8.3.5-py3-none-any.whl (343 kB)
Downloading pluggy-1.6.0-py3-none-any.whl (20 kB)
Downloading iniconfig-2.1.0-py3-none-any.whl (6.0 kB)
Installing collected packages: pluggy, iniconfig, pytest
Successfully installed iniconfig-2.1.0 pluggy-1.6.0 pytest-8.3.5

$ C:\Python313\python.exe -m pip install pyyaml
Defaulting to user installation because normal site-packages is not writeable
Collecting pyyaml
  Using cached PyYAML-6.0.2-cp313-cp313-win_amd64.whl.metadata (2.1 kB)
Using cached PyYAML-6.0.2-cp313-cp313-win_amd64.whl (156 kB)
Installing collected packages: pyyaml
Successfully installed pyyaml-6.0.2

$ C:\Python313\python.exe -m pip install sqlalchemy
Defaulting to user installation because normal site-packages is not writeable
Collecting sqlalchemy
  Using cached sqlalchemy-2.0.41-cp313-cp313-win_amd64.whl.metadata (9.8 kB)
Collecting greenlet>=1 (from sqlalchemy)
  Using cached greenlet-3.2.2-cp313-cp313-win_amd64.whl.metadata (4.2 kB)
Requirement already satisfied: typing-extensions>=4.6.0 in c:\users\packr\appdata\roaming\python\python313\site-packages (from sqlalchemy) (4.13.2)
Using cached sqlalchemy-2.0.41-cp313-cp313-win_amd64.whl (2.1 MB)
Using cached greenlet-3.2.2-cp313-cp313-win_amd64.whl (296 kB)
Installing collected packages: greenlet, sqlalchemy
Successfully installed greenlet-3.2.2 sqlalchemy-2.0.41

```

## 2. 비호환 패키지 설치


### apache-airflow (--pre)
```
$ C:\Python313\python.exe -m pip install --pre apache-airflow
  error: subprocess-exited-with-error
  
  Building wheel for google-re2 (pyproject.toml) did not run successfully.
  exit code: 1
  
  [26 lines of output]
  C:\Users\packr\AppData\Local\Temp\pip-build-env-wkh4l5xf\overlay\Lib\site-packages\setuptools\dist.py:759: SetuptoolsDeprecationWarning: License classifiers are deprecated.
  !!
  
          ********************************************************************************
          Please consider removing the following classifiers in favor of a SPDX license expression:
  
          License :: OSI Approved :: BSD License
  
          See https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#license for details.
          ********************************************************************************
  
  !!
    self._finalize_license_expression()
  running bdist_wheel
  running build
  running build_py
  creating build\lib.win-amd64-cpython-313\re2
  copying re2\__init__.py -> build\lib.win-amd64-cpython-313\re2
  running build_ext
  building '_re2' extension
  creating build\temp.win-amd64-cpython-313\Release
  "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\HostX86\x64\cl.exe" /c /nologo /O2 /W3 /GL /DNDEBUG /MD -IC:\Python313\include -IC:\Python313\Include "-IC:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\include" "-IC:\Program Files (x86)\Windows Kits\10\include\10.0.19041.0\ucrt" "-IC:\Program Files (x86)\Windows Kits\10\include\10.0.19041.0\shared" "-IC:\Program Files (x86)\Windows Kits\10\include\10.0.19041.0\um" "-IC:\Program Files (x86)\Windows Kits\10\include\10.0.19041.0\winrt" "-IC:\Program Files (x86)\Windows Kits\10\include\10.0.19041.0\cppwinrt" /EHsc /Tp_re2.cc /Fobuild\temp.win-amd64-cpython-313\Release\_re2.obj -fvisibility=hidden
  cl : 명령줄 warning D9002 : 알 수 없는 '-fvisibility=hidden' 옵션을 무시합니다.
  _re2.cc
  _re2.cc(15): fatal error C1083: 포함 파일을 열 수 없습니다. 'absl/strings/string_view.h': No such file or directory
  error: command 'C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\VC\\Tools\\MSVC\\14.29.30133\\bin\\HostX86\\x64\\cl.exe' failed with exit code 2
  [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for google-re2

[notice] A new release of pip is available: 25.0.1 -> 25.1.1
[notice] To update, run: python.exe -m pip install --upgrade pip
ERROR: Failed to build installable wheels for some pyproject.toml based projects (google-re2)

```

### apache-airflow (source build)
```
$ C:\Python313\python.exe -m pip install --no-binary :all: --ignore-requires-python apache-airflow
  error: subprocess-exited-with-error
  
  Getting requirements to build wheel did not run successfully.
  exit code: 1
  
  [26 lines of output]
  Traceback (most recent call last):
    File "C:\Python313\Lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 389, in <module>
      main()
      ~~~~^^
    File "C:\Python313\Lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 373, in main
      json_out["return_val"] = hook(**hook_input["kwargs"])
                               ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
    File "C:\Python313\Lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 143, in get_requires_for_build_wheel
      return hook(config_settings)
    File "C:\Users\packr\AppData\Local\Temp\pip-build-env-ulxyjuz8\overlay\Lib\site-packages\setuptools\build_meta.py", line 331, in get_requires_for_build_wheel
      return self._get_build_requires(config_settings, requirements=[])
             ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "C:\Users\packr\AppData\Local\Temp\pip-build-env-ulxyjuz8\overlay\Lib\site-packages\setuptools\build_meta.py", line 301, in _get_build_requires
      self.run_setup()
      ~~~~~~~~~~~~~~^^
    File "C:\Users\packr\AppData\Local\Temp\pip-build-env-ulxyjuz8\overlay\Lib\site-packages\setuptools\build_meta.py", line 317, in run_setup
      exec(code, locals())
      ~~~~^^^^^^^^^^^^^^^^
    File "<string>", line 27, in <module>
    File "C:\Users\packr\AppData\Local\Temp\pip-install-a3q75hon\python-daemon_4b6c67821c8140d08d3eecce527d50d6\util\packaging.py", line 27, in main_module_by_name
      module = __import__(module_name, level=0, fromlist=fromlist)
    File "C:\Users\packr\AppData\Local\Temp\pip-install-a3q75hon\python-daemon_4b6c67821c8140d08d3eecce527d50d6\src\daemon\__init__.py", line 33, in <module>
      from .daemon import DaemonContext
    File "C:\Users\packr\AppData\Local\Temp\pip-install-a3q75hon\python-daemon_4b6c67821c8140d08d3eecce527d50d6\src\daemon\daemon.py", line 13, in <module>
      import pwd
  ModuleNotFoundError: No module named 'pwd'
  [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.

[notice] A new release of pip is available: 25.0.1 -> 25.1.1
[notice] To update, run: python.exe -m pip install --upgrade pip
error: subprocess-exited-with-error

Getting requirements to build wheel did not run successfully.
exit code: 1

See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.

```

### great_expectations (--pre)
```
$ C:\Python313\python.exe -m pip install --pre great_expectations
Defaulting to user installation because normal site-packages is not writeable
Collecting great_expectations
  Downloading great_expectations-1.0.0a4-py3-none-any.whl.metadata (8.7 kB)
Collecting altair<5.0.0,>=4.2.1 (from great_expectations)
  Using cached altair-4.2.2-py3-none-any.whl.metadata (13 kB)
Collecting cryptography>=3.2 (from great_expectations)
  Using cached cryptography-45.0.2-cp311-abi3-win_amd64.whl.metadata (5.7 kB)
Collecting Ipython>=7.16.3 (from great_expectations)
  Using cached ipython-9.2.0-py3-none-any.whl.metadata (4.4 kB)
Collecting ipywidgets>=7.5.1 (from great_expectations)
  Using cached ipywidgets-8.1.7-py3-none-any.whl.metadata (2.4 kB)
Requirement already satisfied: jinja2>=2.10 in c:\users\packr\appdata\roaming\python\python313\site-packages (from great_expectations) (3.1.6)
Collecting jsonschema>=2.5.1 (from great_expectations)
  Using cached jsonschema-4.23.0-py3-none-any.whl.metadata (7.9 kB)
Collecting makefun<2,>=1.7.0 (from great_expectations)
  Using cached makefun-1.16.0-py2.py3-none-any.whl.metadata (2.9 kB)
Collecting marshmallow<4.0.0,>=3.7.1 (from great_expectations)
  Using cached marshmallow-3.26.1-py3-none-any.whl.metadata (7.3 kB)
Collecting mistune>=0.8.4 (from great_expectations)
  Using cached mistune-3.1.3-py3-none-any.whl.metadata (1.8 kB)
Requirement already satisfied: packaging in c:\users\packr\appdata\roaming\python\python313\site-packages (from great_expectations) (25.0)
Collecting posthog<3,>=2.1.0 (from great_expectations)
  Downloading posthog-2.5.0-py2.py3-none-any.whl.metadata (2.0 kB)
Requirement already satisfied: pydantic>=1.10.7 in c:\users\packr\appdata\roaming\python\python313\site-packages (from great_expectations) (2.11.4)
Requirement already satisfied: pyparsing>=2.4 in c:\users\packr\appdata\roaming\python\python313\site-packages (from great_expectations) (3.2.3)
Requirement already satisfied: python-dateutil>=2.8.1 in c:\users\packr\appdata\roaming\python\python313\site-packages (from great_expectations) (2.9.0.post0)
Requirement already satisfied: pytz>=2021.3 in c:\users\packr\appdata\roaming\python\python313\site-packages (from great_expectations) (2025.2)
Requirement already satisfied: requests>=2.20 in c:\users\packr\appdata\roaming\python\python313\site-packages (from great_expectations) (2.32.3)
Collecting ruamel.yaml<0.17.18,>=0.16 (from great_expectations)
  Downloading ruamel.yaml-0.17.17-py3-none-any.whl.metadata (12 kB)
Collecting scipy>=1.6.0 (from great_expectations)
  Using cached scipy-1.15.3-cp313-cp313-win_amd64.whl.metadata (60 kB)
Requirement already satisfied: tqdm>=4.59.0 in c:\users\packr\appdata\roaming\python\python313\site-packages (from great_expectations) (4.67.1)
Requirement already satisfied: typing-extensions>=4.1.0 in c:\users\packr\appdata\roaming\python\python313\site-packages (from great_expectations) (4.13.2)
Collecting tzlocal>=1.2 (from great_expectations)
  Using cached tzlocal-5.3.1-py3-none-any.whl.metadata (7.6 kB)
Requirement already satisfied: urllib3>=1.26 in c:\users\packr\appdata\roaming\python\python313\site-packages (from great_expectations) (2.4.0)
Collecting numpy<2.0.0,>=1.22.4 (from great_expectations)
  Using cached numpy-1.26.4-cp313-cp313-win_amd64.whl
Requirement already satisfied: pandas>=1.3.0 in c:\users\packr\appdata\roaming\python\python313\site-packages (from great_expectations) (2.2.3)
Collecting entrypoints (from altair<5.0.0,>=4.2.1->great_expectations)
  Using cached entrypoints-0.4-py3-none-any.whl.metadata (2.6 kB)
Collecting toolz (from altair<5.0.0,>=4.2.1->great_expectations)
  Using cached toolz-1.0.0-py3-none-any.whl.metadata (5.1 kB)
Collecting cffi>=1.14 (from cryptography>=3.2->great_expectations)
  Using cached cffi-1.17.1-cp313-cp313-win_amd64.whl.metadata (1.6 kB)
Requirement already satisfied: colorama in c:\users\packr\appdata\roaming\python\python313\site-packages (from Ipython>=7.16.3->great_expectations) (0.4.6)
Collecting decorator (from Ipython>=7.16.3->great_expectations)
  Using cached decorator-5.2.1-py3-none-any.whl.metadata (3.9 kB)
Collecting ipython-pygments-lexers (from Ipython>=7.16.3->great_expectations)
  Using cached ipython_pygments_lexers-1.1.1-py3-none-any.whl.metadata (1.1 kB)
Collecting jedi>=0.16 (from Ipython>=7.16.3->great_expectations)
  Using cached jedi-0.19.2-py2.py3-none-any.whl.metadata (22 kB)
Collecting matplotlib-inline (from Ipython>=7.16.3->great_expectations)
  Using cached matplotlib_inline-0.1.7-py3-none-any.whl.metadata (3.9 kB)
Collecting prompt_toolkit<3.1.0,>=3.0.41 (from Ipython>=7.16.3->great_expectations)
  Using cached prompt_toolkit-3.0.51-py3-none-any.whl.metadata (6.4 kB)
Collecting pygments>=2.4.0 (from Ipython>=7.16.3->great_expectations)
  Using cached pygments-2.19.1-py3-none-any.whl.metadata (2.5 kB)
Collecting stack_data (from Ipython>=7.16.3->great_expectations)
  Using cached stack_data-0.6.3-py3-none-any.whl.metadata (18 kB)
Collecting traitlets>=5.13.0 (from Ipython>=7.16.3->great_expectations)
  Using cached traitlets-5.14.3-py3-none-any.whl.metadata (10 kB)
Collecting comm>=0.1.3 (from ipywidgets>=7.5.1->great_expectations)
  Using cached comm-0.2.2-py3-none-any.whl.metadata (3.7 kB)
Collecting widgetsnbextension~=4.0.14 (from ipywidgets>=7.5.1->great_expectations)
  Using cached widgetsnbextension-4.0.14-py3-none-any.whl.metadata (1.6 kB)
Collecting jupyterlab_widgets~=3.0.15 (from ipywidgets>=7.5.1->great_expectations)
  Using cached jupyterlab_widgets-3.0.15-py3-none-any.whl.metadata (20 kB)
Requirement already satisfied: MarkupSafe>=2.0 in c:\users\packr\appdata\roaming\python\python313\site-packages (from jinja2>=2.10->great_expectations) (3.0.2)
Collecting attrs>=22.2.0 (from jsonschema>=2.5.1->great_expectations)
  Using cached attrs-25.3.0-py3-none-any.whl.metadata (10 kB)
Collecting jsonschema-specifications>=2023.03.6 (from jsonschema>=2.5.1->great_expectations)
  Using cached jsonschema_specifications-2025.4.1-py3-none-any.whl.metadata (2.9 kB)
Collecting referencing>=0.28.4 (from jsonschema>=2.5.1->great_expectations)
  Using cached referencing-0.36.2-py3-none-any.whl.metadata (2.8 kB)
Collecting rpds-py>=0.7.1 (from jsonschema>=2.5.1->great_expectations)
  Using cached rpds_py-0.25.0-cp313-cp313-win_amd64.whl.metadata (4.2 kB)
Requirement already satisfied: tzdata>=2022.7 in c:\users\packr\appdata\roaming\python\python313\site-packages (from pandas>=1.3.0->great_expectations) (2025.2)
Requirement already satisfied: six>=1.5 in c:\users\packr\appdata\roaming\python\python313\site-packages (from posthog<3,>=2.1.0->great_expectations) (1.17.0)
Collecting monotonic>=1.5 (from posthog<3,>=2.1.0->great_expectations)
  Downloading monotonic-1.6-py2.py3-none-any.whl.metadata (1.5 kB)
Collecting backoff>=1.10.0 (from posthog<3,>=2.1.0->great_expectations)
  Using cached backoff-2.2.1-py3-none-any.whl.metadata (14 kB)
Requirement already satisfied: annotated-types>=0.6.0 in c:\users\packr\appdata\roaming\python\python313\site-packages (from pydantic>=1.10.7->great_expectations) (0.7.0)
Requirement already satisfied: pydantic-core==2.33.2 in c:\users\packr\appdata\roaming\python\python313\site-packages (from pydantic>=1.10.7->great_expectations) (2.33.2)
Requirement already satisfied: typing-inspection>=0.4.0 in c:\users\packr\appdata\roaming\python\python313\site-packages (from pydantic>=1.10.7->great_expectations) (0.4.0)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\packr\appdata\roaming\python\python313\site-packages (from requests>=2.20->great_expectations) (3.4.2)
Requirement already satisfied: idna<4,>=2.5 in c:\users\packr\appdata\roaming\python\python313\site-packages (from requests>=2.20->great_expectations) (3.10)
Requirement already satisfied: certifi>=2017.4.17 in c:\users\packr\appdata\roaming\python\python313\site-packages (from requests>=2.20->great_expectations) (2025.4.26)
Collecting pycparser (from cffi>=1.14->cryptography>=3.2->great_expectations)
  Using cached pycparser-2.22-py3-none-any.whl.metadata (943 bytes)
Collecting parso<0.9.0,>=0.8.4 (from jedi>=0.16->Ipython>=7.16.3->great_expectations)
  Using cached parso-0.8.4-py2.py3-none-any.whl.metadata (7.7 kB)
Collecting wcwidth (from prompt_toolkit<3.1.0,>=3.0.41->Ipython>=7.16.3->great_expectations)
  Using cached wcwidth-0.2.13-py2.py3-none-any.whl.metadata (14 kB)
Collecting executing>=1.2.0 (from stack_data->Ipython>=7.16.3->great_expectations)
  Using cached executing-2.2.0-py2.py3-none-any.whl.metadata (8.9 kB)
Collecting asttokens>=2.1.0 (from stack_data->Ipython>=7.16.3->great_expectations)
  Using cached asttokens-3.0.0-py3-none-any.whl.metadata (4.7 kB)
Collecting pure-eval (from stack_data->Ipython>=7.16.3->great_expectations)
  Using cached pure_eval-0.2.3-py3-none-any.whl.metadata (6.3 kB)
Downloading great_expectations-1.0.0a4-py3-none-any.whl (4.9 MB)
   ---------------------------------------- 4.9/4.9 MB 6.2 MB/s eta 0:00:00
Using cached altair-4.2.2-py3-none-any.whl (813 kB)
Using cached cryptography-45.0.2-cp311-abi3-win_amd64.whl (3.4 MB)
Using cached ipython-9.2.0-py3-none-any.whl (604 kB)
Using cached ipywidgets-8.1.7-py3-none-any.whl (139 kB)
Using cached jsonschema-4.23.0-py3-none-any.whl (88 kB)
Using cached makefun-1.16.0-py2.py3-none-any.whl (22 kB)
Using cached marshmallow-3.26.1-py3-none-any.whl (50 kB)
Using cached mistune-3.1.3-py3-none-any.whl (53 kB)
Downloading posthog-2.5.0-py2.py3-none-any.whl (36 kB)
Downloading ruamel.yaml-0.17.17-py3-none-any.whl (109 kB)
Using cached scipy-1.15.3-cp313-cp313-win_amd64.whl (41.0 MB)
Using cached tzlocal-5.3.1-py3-none-any.whl (18 kB)
Using cached attrs-25.3.0-py3-none-any.whl (63 kB)
Using cached backoff-2.2.1-py3-none-any.whl (15 kB)
Using cached cffi-1.17.1-cp313-cp313-win_amd64.whl (182 kB)
Using cached comm-0.2.2-py3-none-any.whl (7.2 kB)
Using cached jedi-0.19.2-py2.py3-none-any.whl (1.6 MB)
Using cached jsonschema_specifications-2025.4.1-py3-none-any.whl (18 kB)
Using cached jupyterlab_widgets-3.0.15-py3-none-any.whl (216 kB)
Downloading monotonic-1.6-py2.py3-none-any.whl (8.2 kB)
Using cached prompt_toolkit-3.0.51-py3-none-any.whl (387 kB)
Using cached pygments-2.19.1-py3-none-any.whl (1.2 MB)
Using cached referencing-0.36.2-py3-none-any.whl (26 kB)
Using cached rpds_py-0.25.0-cp313-cp313-win_amd64.whl (234 kB)
Using cached traitlets-5.14.3-py3-none-any.whl (85 kB)
Using cached widgetsnbextension-4.0.14-py3-none-any.whl (2.2 MB)
Using cached decorator-5.2.1-py3-none-any.whl (9.2 kB)
Using cached entrypoints-0.4-py3-none-any.whl (5.3 kB)
Using cached ipython_pygments_lexers-1.1.1-py3-none-any.whl (8.1 kB)
Using cached matplotlib_inline-0.1.7-py3-none-any.whl (9.9 kB)
Using cached stack_data-0.6.3-py3-none-any.whl (24 kB)
Using cached toolz-1.0.0-py3-none-any.whl (56 kB)
Using cached asttokens-3.0.0-py3-none-any.whl (26 kB)
Using cached executing-2.2.0-py2.py3-none-any.whl (26 kB)
Using cached parso-0.8.4-py2.py3-none-any.whl (103 kB)
Using cached pure_eval-0.2.3-py3-none-any.whl (11 kB)
Using cached pycparser-2.22-py3-none-any.whl (117 kB)
Using cached wcwidth-0.2.13-py2.py3-none-any.whl (34 kB)
Installing collected packages: wcwidth, pure-eval, monotonic, makefun, widgetsnbextension, tzlocal, traitlets, toolz, ruamel.yaml, rpds-py, pygments, pycparser, prompt_toolkit, parso, numpy, mistune, marshmallow, jupyterlab_widgets, executing, entrypoints, decorator, backoff, attrs, asttokens, stack_data, scipy, referencing, posthog, matplotlib-inline, jedi, ipython-pygments-lexers, comm, cffi, jsonschema-specifications, Ipython, cryptography, jsonschema, ipywidgets, altair, great_expectations
  Attempting uninstall: numpy
    Found existing installation: numpy 2.2.6
    Uninstalling numpy-2.2.6:
      Successfully uninstalled numpy-2.2.6
Successfully installed Ipython-9.2.0 altair-4.2.2 asttokens-3.0.0 attrs-25.3.0 backoff-2.2.1 cffi-1.17.1 comm-0.2.2 cryptography-45.0.2 decorator-5.2.1 entrypoints-0.4 executing-2.2.0 great_expectations-1.0.0a4 ipython-pygments-lexers-1.1.1 ipywidgets-8.1.7 jedi-0.19.2 jsonschema-4.23.0 jsonschema-specifications-2025.4.1 jupyterlab_widgets-3.0.15 makefun-1.16.0 marshmallow-3.26.1 matplotlib-inline-0.1.7 mistune-3.1.3 monotonic-1.6 numpy-1.26.4 parso-0.8.4 posthog-2.5.0 prompt_toolkit-3.0.51 pure-eval-0.2.3 pycparser-2.22 pygments-2.19.1 referencing-0.36.2 rpds-py-0.25.0 ruamel.yaml-0.17.17 scipy-1.15.3 stack_data-0.6.3 toolz-1.0.0 traitlets-5.14.3 tzlocal-5.3.1 wcwidth-0.2.13 widgetsnbextension-4.0.14

```
Note: markdownlint-cli2는 npm 패키지입니다. 다음 명령어로 설치하세요:
```
npm install -g markdownlint-cli2
```
