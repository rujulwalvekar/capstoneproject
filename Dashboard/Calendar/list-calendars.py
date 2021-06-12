from pprint import pprint
from Google import Create_Service

CLIENT_SECRET_FILE = 'client_secret.json'
API_NAME = 'calendar'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/calendar']

service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

response = service.calendarList().list().execute()
pprint(response)

# print(response.keys())
# print("ITEMS ", response.get('items')[0])

calendarItems = response.get('items')
nextPageToken = response.get('nextPageToken')

while nextPageToken:
    response = service.calendarsList().list(
        maxResults=250,
        showDeleted=False,
        showHidden=False,
        pageToken=nextPageToken
    ).execute()
    calendarItems.extend(response.get('items'))
    nextPageToken = response.get('nextPageToken')



