import yt_dlp
import sys

def download_video(video_url, ydl_opts, video_number, total_videos):
    print(f"Downloading item {video_number} of {total_videos}: {video_url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

def get_video_urls(channel_url):
    video_urls = []
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(channel_url, download=False)
        if 'entries' in result:
            for entry in result['entries']:
                if 'url' in entry:
                    video_urls.append(entry['url'])
    return video_urls

def download_all_videos_from_channel(channel_url, start_from=1):
    video_urls = get_video_urls(channel_url)
    total_videos = len(video_urls)
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',  # Default video quality with best audio
        'outtmpl': '%(title)s [%(id)s].%(ext)s',
        'noplaylist': False,
        'download_archive': 'downloaded_videos.txt',
    }

    for i, url in enumerate(video_urls[start_from-1:], start_from):
        download_video(url, ydl_opts, i, total_videos)

if __name__ == "__main__":
    start_from = 1
    channel_url = "https://www.youtube.com/@internationalcentreforsett919/videos"  # Default URL

    if len(sys.argv) > 1:
        channel_url = sys.argv[1]
        if len(sys.argv) > 2:
            start_from = int(sys.argv[2])

    download_all_videos_from_channel(channel_url, start_from)
