# YouTube API 및 Cloud Video Intelligence API 구성 파일
# 이 파일은 API 키와 서비스 계정 설정 정보를 포함합니다

# YouTube Data API v3 설정
YOUTUBE_API_KEY = "SAMPLE_YOUTUBE_API_KEY"  # 이 샘플 키는 실제 API 키로 대체해야 합니다

# Google Cloud 설정
PROJECT_ID = "youtube-ai-integration"  # 실제 프로젝트 ID로 대체하세요
GCS_BUCKET = "youtube-videos-storage"  # 비디오 저장용 Cloud Storage 버킷 이름

# 서비스 계정 설정
SERVICE_ACCOUNT_FILE = "C:\\Users\\packr\\OneDrive\\palantir\\config\\service-account.json"

# 임시 파일 저장 경로
TEMP_DOWNLOAD_DIR = "C:\\Users\\packr\\OneDrive\\palantir\\temp\\youtube"

# 분석 결과 저장 경로
ANALYSIS_OUTPUT_DIR = "C:\\Users\\packr\\OneDrive\\palantir\\output\\youtube"
