## Import Libraries 

import openai
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from googleapiclient.discovery import build

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime

import whisper
from gtts import gTTS
import datetime
import re

## Set API Keys

os.environ["OPENAI_API_KEY"] = "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"
os.environ["YOUTUBE_API_KEY"] = "AIzaSyD1erYGrTCwx6wLd4s_X_4AsU6LeqNe1XI"
openai.api_key = os.environ["OPENAI_API_KEY"]
YOUTUBE_API_KEY = os.environ["YOUTUBE_API_KEY"]

## User Input

# channel_id = input("Enter your YouTube channel ID: ")
# video_idea = input("Enter your next video idea: ")
# style_input = input("Choose title style - 1. Conservative / 2. Bold / 3. Trendy: ")

# Map style
# style_map = {"1": "Conservative", "2": "Bold", "3": "Trendy"}
# title_style = style_map.get(style_input, "Trendy")

# Tools

## Channel Name
@tool
def get_channel_name(channel_id: str) -> str:
    """Fetches and displays the YouTube channel name from channel ID."""
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        channel_response = youtube.channels().list(
            part="snippet",
            id=channel_id
        ).execute()

        if channel_response.get("items"):
            channel_name = channel_response["items"][0]["snippet"]["title"]
            return f"📺 Channel: {channel_name}"
        else:
            return "❌ Channel not found. Please check your channel ID."

    except Exception as e:
        return f"❌ Error fetching channel name: {str(e)}"
    
## Title
@tool
def generate_title(video_idea: str, style: str) -> str:
    """Generates three video titles for a YouTube idea based on selected style."""
    prompt = f"""
You are a YouTube expert.
Suggest 3 catchy titles in {style} style for this video idea: "{video_idea}".
"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
    title_suggestions = llm.invoke(prompt).content
    return title_suggestions

## Description
@tool
def generate_description(video_idea: str, style: str) -> str:
    """Generates a short video description for YouTube content."""
    prompt = f"""
Create a short engaging YouTube description in {style} style for the topic: "{video_idea}".
Limit to 1-2 sentences.
"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
    return llm.invoke(prompt).content

## Scheduler Advisor
@tool
def video_scheduler_advisor(channel_id: str, video_topic: str) -> str:
    """Suggests the best day and time to post a video based on channel's historical performance."""

    def fetch_youtube_posting_data(channel_id, api_key, max_results=50):
        """Fetch posting times data for analysis."""
        try:
            youtube = build("youtube", "v3", developerKey=api_key)

            # Get channel uploads playlist
            channel_response = youtube.channels().list(
                part="contentDetails",
                id=channel_id
            ).execute()

            if not channel_response.get("items"):
                return None

            uploads_playlist = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
            videos = []
            next_page_token = None

            # Get video list with stats
            while len(videos) < max_results:
                playlist_response = youtube.playlistItems().list(
                    part="snippet",
                    playlistId=uploads_playlist,
                    maxResults=min(25, max_results - len(videos)),
                    pageToken=next_page_token
                ).execute()

                video_ids = []
                for item in playlist_response["items"]:
                    video_ids.append(item["snippet"]["resourceId"]["videoId"])
                    videos.append({
                        "video_id": item["snippet"]["resourceId"]["videoId"],
                        "title": item["snippet"]["title"],
                        "published_at": item["snippet"]["publishedAt"]
                    })

                # Get detailed stats
                if video_ids:
                    video_details = youtube.videos().list(
                        part="statistics",
                        id=",".join(video_ids)
                    ).execute()

                    # Add stats to videos
                    for i, video_detail in enumerate(video_details.get("items", [])):
                        if i < len(videos):
                            stats = video_detail.get("statistics", {})
                            videos[-(len(video_ids)-i)]["view_count"] = int(stats.get("viewCount", 0))
                            videos[-(len(video_ids)-i)]["like_count"] = int(stats.get("likeCount", 0))
                            videos[-(len(video_ids)-i)]["engagement_rate"] = (
                                int(stats.get("likeCount", 0)) + int(stats.get("commentCount", 0))
                            ) / max(int(stats.get("viewCount", 1)), 1) * 100  # Prevent division by zero

                next_page_token = playlist_response.get("nextPageToken")
                if not next_page_token:
                    break

            return pd.DataFrame(videos)

        except Exception as e:
            print(f"Error fetching scheduling data: {e}")
            return None

    def analyze_best_posting_times(df):
        """Analyze historical data to find best posting patterns."""
        if df is None or df.empty:
            return None, None, None

        # Process datetime
        df["published_at"] = pd.to_datetime(df["published_at"])
        df["day_name"] = df["published_at"].dt.day_name()
        df["hour"] = df["published_at"].dt.hour
        df["day_num"] = df["published_at"].dt.dayofweek  # 0=Monday, 6=Sunday

        # Calculate performance scores (views + engagement)
        df["performance_score"] = df.get("view_count", 0) + (df.get("engagement_rate", 0) * 100)

        # Find best day based on average performance
        day_performance = df.groupby("day_name").agg({
            "performance_score": "mean",
            "view_count": "mean",
            "video_id": "count"  # Number of videos posted
        }).round(2)

        # Find best hour based on average performance
        hour_performance = df.groupby("hour").agg({
            "performance_score": "mean",
            "view_count": "mean",
            "video_id": "count"
        }).round(2)

        # Get top recommendations
        best_day = day_performance["performance_score"].idxmax()
        best_hour = hour_performance["performance_score"].idxmax()

        # Get additional insights
        total_videos = len(df)
        avg_views = df["view_count"].mean()

        return best_day, best_hour, {
            "day_performance": day_performance,
            "hour_performance": hour_performance,
            "total_videos": total_videos,
            "avg_views": avg_views,
            "best_day_avg_views": day_performance.loc[best_day, "view_count"],
            "best_hour_avg_views": hour_performance.loc[best_hour, "view_count"]
        }

    def get_topic_specific_advice(video_topic, best_day, best_hour, stats):
        """Use AI to give topic-specific scheduling advice."""
        try:
            prompt = f"""
You are a YouTube scheduling expert. Based on the data analysis, provide specific advice for posting this video.

Video Topic: "{video_topic}"
Best Historical Day: {best_day}
Best Historical Hour: {best_hour}:00
Channel Average Views: {stats['avg_views']:,.0f}
Best Day Average Views: {stats['best_day_avg_views']:,.0f}
Total Videos Analyzed: {stats['total_videos']}

Provide a short, actionable recommendation that includes:
1. Specific day and time suggestion
2. Brief reason why this timing works
3. One bonus tip for this topic

Keep it conversational and under 100 words.
"""

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
            response = llm.invoke(prompt).content

            return response

        except Exception as e:
            return f"Based on your channel data, post on **{best_day} at {best_hour}:00** for optimal performance!"

    # Main execution
    print("📊 Analyzing your posting patterns...")

    # Fetch and analyze data
    df = fetch_youtube_posting_data(channel_id, YOUTUBE_API_KEY)
    best_day, best_hour, stats = analyze_best_posting_times(df)

    if not best_day or not best_hour:
        return "⚠️ Could not analyze posting patterns. Please check your channel ID or ensure you have enough videos."

    # Get AI-powered advice
    advice = get_topic_specific_advice(video_topic, best_day, best_hour, stats)

    # Create comprehensive recommendation
    recommendation = f"""
🎯 VIDEO SCHEDULER ADVISOR RESULTS:

📅 **RECOMMENDED POSTING TIME:**
{advice}

📈 **DATA INSIGHTS:**
• Best performing day: **{best_day}** (avg {stats['best_day_avg_views']:,.0f} views)
• Best performing hour: **{best_hour}:00** (avg {stats['best_hour_avg_views']:,.0f} views)
• Analysis based on {stats['total_videos']} videos
• Your channel average: {stats['avg_views']:,.0f} views per video

⏰ **QUICK STATS:**
• Most active posting days: {', '.join(stats['day_performance'].sort_values('video_id', ascending=False).head(3).index.tolist())}
• Most active posting hours: {', '.join([f"{h}:00" for h in stats['hour_performance'].sort_values('video_id', ascending=False).head(3).index.tolist()])}

💡 **SCHEDULING TIP:**
Videos posted on {best_day} at {best_hour}:00 perform {((stats['best_day_avg_views'] / stats['avg_views'] - 1) * 100):+.1f}% better than your average!
"""

    return recommendation

## Content Classification & Interactive Visualizations
@tool
def classify_and_visualize_content_interactive(channel_id: str) -> str:
    """Creates interactive YouTube channel analytics dashboard using Plotly."""

    def fetch_youtube_video_data_with_descriptions(channel_id, api_key, max_results=30):
        """Fetch video data including descriptions for better classification."""
        try:
            youtube = build("youtube", "v3", developerKey=api_key)
            # Get channel uploads playlist
            channel_response = youtube.channels().list(
                part="contentDetails",
                id=channel_id
            ).execute()
            if not channel_response.get("items"):
                return None
            uploads_playlist = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
            videos = []
            next_page_token = None
            # Get video list from playlist
            while len(videos) < max_results:
                playlist_response = youtube.playlistItems().list(
                    part="snippet",
                    playlistId=uploads_playlist,
                    maxResults=min(25, max_results - len(videos)),
                    pageToken=next_page_token
                ).execute()
                video_ids = []
                for item in playlist_response["items"]:
                    video_ids.append(item["snippet"]["resourceId"]["videoId"])
                    videos.append({
                        "video_id": item["snippet"]["resourceId"]["videoId"],
                        "title": item["snippet"]["title"],
                        "published_at": item["snippet"]["publishedAt"],
                        "description": ""  # Will be filled below
                    })
                # Get detailed video information including descriptions
                if video_ids:
                    video_details = youtube.videos().list(
                        part="snippet,statistics",  # Added statistics for view counts
                        id=",".join(video_ids)
                    ).execute()

                    # Match descriptions and stats to videos
                    for i, video_detail in enumerate(video_details.get("items", [])):
                        if i < len(videos):
                            videos[-(len(video_ids)-i)]["description"] = video_detail["snippet"].get("description", "")[:500]
                            # Add view count if available
                            stats = video_detail.get("statistics", {})
                            videos[-(len(video_ids)-i)]["view_count"] = int(stats.get("viewCount", 0))
                            videos[-(len(video_ids)-i)]["like_count"] = int(stats.get("likeCount", 0))
                            videos[-(len(video_ids)-i)]["comment_count"] = int(stats.get("commentCount", 0))
                next_page_token = playlist_response.get("nextPageToken")
                if not next_page_token:
                    break
            return pd.DataFrame(videos)
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def ai_classify_content(title, description, batch_size=5):
        """Use OpenAI to intelligently classify content with multi-language support."""
        try:
            content_text = f"Title: {title}\nDescription: {description[:200]}..."

            prompt = f"""
You are an expert content classifier for YouTube videos. Classify the following video content into ONE of these categories:
Categories:
1. Sports (Football/Soccer, Basketball, Tennis, Cricket, Racing, Olympics, etc.)
2. News/Analysis (Breaking news, sports analysis, interviews, reports)
3. Entertainment (Comedy, reactions, challenges, funny moments)
4. Educational/Tutorial (How-to guides, explanations, learning content)
5. Gaming (Video games, esports, gaming reviews)
6. Lifestyle (Daily vlogs, personal content, lifestyle tips)
7. Music (Songs, concerts, music videos, covers)
8. Technology (Tech reviews, gadgets, software)
9. Travel (Travel vlogs, destinations, tourism)
10. Other (Content that doesn't fit above categories)

Content to classify (may be in Arabic, English, or other languages):
{content_text}

Instructions:
- Analyze both title and description
- Consider the language (Arabic, English, etc.)
- Focus on the main topic/theme
- Return ONLY the category name (e.g., "Sports" or "News/Analysis")
- If it's clearly about sports but in Arabic, still classify as "Sports"

Category:"""
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.1
            )

            classification = response.choices[0].message.content.strip()

            valid_categories = [
                "Sports", "News/Analysis", "Entertainment", "Educational/Tutorial",
                "Gaming", "Lifestyle", "Music", "Technology", "Travel", "Other"
            ]

            return classification if classification in valid_categories else "Other"

        except Exception as e:
            print(f"AI Classification error: {e}")
            return "Other"

    def batch_classify_content(df):
        """Classify content in batches to be more efficient with API calls."""
        classifications = []

        print("🤖 Using AI to classify content... This may take a moment.")

        for index, row in df.iterrows():
            classification = ai_classify_content(row['title'], row['description'])
            classifications.append(classification)
            print(f"Classified {index+1}/{len(df)}: {classification}")

            import time
            time.sleep(0.5)

        return classifications

    # Fetch data with descriptions
    print("📊 Fetching YouTube data with descriptions...")
    df = fetch_youtube_video_data_with_descriptions(channel_id, YOUTUBE_API_KEY)
    if df is None or df.empty:
        return "Could not fetch data. Please check your channel ID."

    # Process datetime data
    df["published_at"] = pd.to_datetime(df["published_at"])
    df["day"] = df["published_at"].dt.day_name()
    df["hour"] = df["published_at"].dt.hour
    df["date"] = df["published_at"].dt.date

    # AI-powered content classification
    df["content_type"] = batch_classify_content(df)

    # CREATE INTERACTIVE PLOTLY DASHBOARD

    # Color schemes
# FIND THIS SECTION IN YOUR CODE (around line 130-250 in the interactive function)
# REPLACE the entire "CREATE INTERACTIVE PLOTLY DASHBOARD" section with this:

    # CREATE INTERACTIVE PLOTLY DASHBOARD

    # Better color schemes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
              '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43']

    # Create subplots with white background
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('📅 Best Posting Times Heatmap',
                       '⏰ Hourly Posting Pattern',
                       '📊 Content Distribution',
                       '📈 Performance Timeline'),
        specs=[[{"type": "heatmap"}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "scatter"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # 1. BEAUTIFUL HEATMAP - Posting Times with rounded corners effect
    heatmap_data = df.groupby(["day", "hour"]).size().unstack(fill_value=0)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order, fill_value=0)

    if not heatmap_data.empty:
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=list(heatmap_data.columns),
                y=day_order,
                colorscale='Plasma',  # Beautiful plasma colors back!
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>Hour: %{x}<br>Videos Posted: %{z}<extra></extra>',
                showscale=True,
                colorbar=dict(
                    x=0.48,
                    len=0.4,
                    y=0.75,
                    bgcolor='white',
                    bordercolor='gray',
                    borderwidth=1
                ),
                # Add gaps between squares for cleaner look
                xgap=2,
                ygap=2
            ),
            row=1, col=1
        )

    # 2. COLORFUL BAR CHART - Hourly Distribution
    hour_counts = df['hour'].value_counts().sort_index()
    if not hour_counts.empty:
        # Create gradient colors for bars
        bar_colors = [f'rgba({int(255*(i/len(hour_counts)))}, {int(100+155*(1-i/len(hour_counts)))}, {int(200+55*(i/len(hour_counts)))}, 0.8)'
                     for i in range(len(hour_counts))]

        fig.add_trace(
            go.Bar(
                x=hour_counts.index,
                y=hour_counts.values,
                marker=dict(
                    color=bar_colors,
                    line=dict(color='rgba(0,0,0,0.2)', width=1)
                ),
                hovertemplate='<b>Hour %{x}:00</b><br>Videos: %{y}<extra></extra>',
                name="Hourly Posts"
            ),
            row=1, col=2
        )

    # 3. VIBRANT PIE CHART - Content Distribution
    content_counts = df['content_type'].value_counts()
    if not content_counts.empty:
        fig.add_trace(
            go.Pie(
                labels=content_counts.index,
                values=content_counts.values,
                marker=dict(
                    colors=colors[:len(content_counts)],
                    line=dict(color='white', width=2)  # White borders for clarity
                ),
                hovertemplate='<b>%{label}</b><br>Videos: %{value}<br>Percentage: %{percent}<extra></extra>',
                textinfo='label+percent',
                textposition='inside',
                textfont=dict(size=12, color='white')
            ),
            row=2, col=1
        )

    # 4. CLEAN TIMELINE - Performance over time
    if 'view_count' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['published_at'],
                y=df['view_count'],
                mode='markers+lines',
                marker=dict(
                    size=12,
                    color=df['view_count'],
                    colorscale='Plasma',  # Plasma colors here too!
                    showscale=True,
                    colorbar=dict(
                        x=1.02,
                        len=0.4,
                        y=0.25,
                        bgcolor='white',
                        bordercolor='gray',
                        borderwidth=1
                    ),
                    line=dict(color='white', width=2)  # White border around markers
                ),
                line=dict(width=3, color='rgba(68, 68, 68, 0.8)'),
                hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Views: %{y:,}<extra></extra>',
                text=df['title'].str[:30] + '...',
                name="Video Performance"
            ),
            row=2, col=2
        )
    else:
        # Fallback: colorful posting frequency over time
        daily_posts = df.groupby('date').size().reset_index()
        daily_posts.columns = ['date', 'posts']
        fig.add_trace(
            go.Scatter(
                x=daily_posts['date'],
                y=daily_posts['posts'],
                mode='markers+lines',
                marker=dict(
                    size=10,
                    color='#FF6B6B',
                    line=dict(color='white', width=2)
                ),
                line=dict(width=3, color='#FF6B6B'),
                hovertemplate='<b>Date: %{x}</b><br>Posts: %{y}<extra></extra>',
                name="Daily Posts"
            ),
            row=2, col=2
        )

    # PERFECT LAYOUT - White background and beautiful styling
    fig.update_layout(
        height=800,
        title_text="🚀 Interactive YouTube Channel Analytics Dashboard",
        title_x=0.5,
        title_font=dict(size=22, color='black'),
        showlegend=False,
        font=dict(size=12, color='black'),
        plot_bgcolor='white',          # WHITE BACKGROUND!
        paper_bgcolor='white',         # WHITE PAPER!
        margin=dict(l=50, r=50, t=80, b=50)
    )

    # Clean white grid lines
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.3)',
        linecolor='gray',
        showline=True
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.3)',
        linecolor='gray',
        showline=True
    )

    # Make subplot titles more visible
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=14, color='black')

    # Show the interactive plot
    # fig.show()

    # Also save as HTML for sharing
    fig.write_html("interactive_youtube_analytics.html")
    print("💾 Interactive dashboard saved as 'interactive_youtube_analytics.html'")

    # CREATE BONUS: BEAUTIFUL Content Performance Chart
    if not df.empty:
        performance_fig = go.Figure()

        for i, content_type in enumerate(df['content_type'].unique()):
            df_type = df[df['content_type'] == content_type]

            performance_fig.add_trace(
                go.Box(
                    y=df_type.get('view_count', [0] * len(df_type)),
                    name=content_type,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8,
                    marker=dict(color=colors[i % len(colors)]),
                    line=dict(color=colors[i % len(colors)]),
                    hovertemplate=f'<b>{content_type}</b><br>Views: %{{y:,}}<extra></extra>'
                )
            )

        performance_fig.update_layout(
            title=dict(
                text="📈 Content Performance by Category",
                font=dict(size=20, color='black')
            ),
            yaxis_title="View Count",
            xaxis_title="Content Type",
            height=500,
            showlegend=False,
            plot_bgcolor='white',          # WHITE BACKGROUND!
            paper_bgcolor='white',         # WHITE PAPER!
            font=dict(color='black'),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.3)',
                linecolor='gray'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.3)',
                linecolor='gray'
            )
        )

        # performance_fig.show()
        performance_fig.write_html("content_performance_analysis.html")
        print("💾 Performance analysis saved as 'content_performance_analysis.html'")

    # Generate insights
    if not content_counts.empty:
        most_common_content = content_counts.index[0]
        most_common_count = content_counts.iloc[0]
    else:
        most_common_content = "Unknown"
        most_common_count = 0

    if not hour_counts.empty:
        best_hour = hour_counts.idxmax()
    else:
        best_hour = "Unknown"

    total_videos = len(df)
    avg_views = df.get('view_count', [0]).mean() if 'view_count' in df.columns else 0

    content_breakdown = "\n".join([f"• {cat}: {count} videos ({count/total_videos*100:.1f}%)"
                                  for cat, count in content_counts.items()])

    insights = f"""
🚀 INTERACTIVE AI-POWERED CHANNEL INSIGHTS:
• Total videos analyzed: {total_videos}
• Most common content type: {most_common_content} ({most_common_count} videos)
• Best posting hour: {best_hour}:00
• Average views per video: {avg_views:,.0f}
• Content diversity: {len(content_counts)} different categories

📊 DETAILED CONTENT BREAKDOWN:
{content_breakdown}

🎯 RECOMMENDATIONS:
• Post more content during hour {best_hour}:00 (your best performing time)
• Consider diversifying into {content_counts.index[-1] if len(content_counts) > 1 else 'new categories'}
• Focus on {most_common_content} content (your strongest category)
    """

    return f"✅ Interactive analytics dashboard created! {insights}"

## Thumbnail Generator & Cost Tracking
@tool
def generate_smart_thumbnail_with_cost_tracking(video_idea: str, video_title: str) -> str:
    """
    Smart thumbnail generator with cost tracking.
    Perfect for junior developers - simple, effective, and budget-aware!
    """

    # Simple cost calculator
    def calculate_thumbnail_cost():
        """Calculate the cost of generating one thumbnail."""
        # DALL-E 3 pricing (as of 2024)
        DALLE3_COST_PER_IMAGE = 0.040  # Standard quality, 1024x1024
        return DALLE3_COST_PER_IMAGE

    def create_super_smart_prompt(video_idea, video_title):
        """
        Super smart prompt - let DALL-E figure out everything!
        Perfect for junior developers: simple but powerful.
        """

        smart_prompt = f"""
Professional YouTube thumbnail for: "{video_title}"
Topic: {video_idea}

Visual requirements:
- Vibrant, eye-catching colors with maximum contrast
- Bold, readable text that pops on mobile screens
- Clean, modern design with dramatic professional lighting
- Engaging visual elements directly related to "{video_idea}"
- Perfect 16:9 YouTube aspect ratio (1024x1024 cropped works fine)
- High-quality photorealistic style with sharp details
- Dynamic gradient background that complements the topic
- Strategic composition optimized for click-through rates
- Text placement that doesn't obscure key visual elements
- Mobile-first design approach for small screen readability
- Content-appropriate styling (let AI understand from context)

Style: Professional, attention-grabbing, modern, YouTube-optimized for maximum clicks.
"""

        return smart_prompt.strip()

    def generate_thumbnail_with_cost_awareness(prompt):
        """Generate thumbnail and track costs."""
        estimated_cost = calculate_thumbnail_cost()

        print(f"💰 Estimated cost: ${estimated_cost:.3f}")
        print("🎨 Generating thumbnail...")

        try:
            response = openai.images.generate(
                prompt=prompt,
                model="dall-e-3",
                size="1024x1024",
                quality="standard",
                n=1
            )

            image_url = response.data[0].url
            # display(Image(url=image_url))

            return image_url, estimated_cost

        # except Exception as e:
        #     return f"❌ Error: {str(e)}", 0
        except Exception as e:
            return f"❌ Thumbnail generation failed: {e}", 0


    # Main execution
    print("🎨 Creating smart thumbnail with cost tracking...")

    # Calculate cost upfront
    estimated_cost = calculate_thumbnail_cost()
    print(f"💰 This will cost approximately: ${estimated_cost:.3f}")

    # Create the smart prompt
    smart_prompt = create_super_smart_prompt(video_idea, video_title)

    # Generate thumbnail with cost tracking
    result, actual_cost = generate_thumbnail_with_cost_awareness(smart_prompt)

    if "Error" not in str(result):
        return f"""
✅ SMART THUMBNAIL GENERATED WITH COST TRACKING!

🎨 **CONTENT DETAILS:**
• Topic: {video_idea}
• Title: {video_title}

🔗 **THUMBNAIL URL:** {result}

📸 **THUMBNAIL PREVIEW:**
<img src="{result}" alt="Generated Thumbnail" style="max-width: 512px; height: auto; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">

💡 **SMART FEATURES:**
• DALL-E automatically understands content type from your input
• Professional prompt engineering optimized for thumbnails
• YouTube-ready 16:9 aspect ratio
• Mobile-optimized high contrast design
• Zero complexity - perfect for junior developers!

💰 **COST BREAKDOWN:**
• DALL-E 3 Standard Quality: ${actual_cost:.3f}
• Total Cost: ${actual_cost:.3f}
• No extra API calls = Maximum cost efficiency!

📊 **JUNIOR DEV BENEFITS:**
• Simple one-function approach
• No complex AI classification needed
• Easy to debug and maintain
• Cost-transparent
• Production-ready results

🚀 **READY TO USE!**
        """
    else:
        return f"❌ Generation failed. Cost: $0.00\nError: {result}"

# Optional: Add a simple budget tracker
class SimpleBudgetTracker:
    """Simple budget tracker for junior developers."""

    def __init__(self, monthly_budget=10.0):
        self.monthly_budget = monthly_budget
        self.spent_this_month = 0.0

    def add_expense(self, cost):
        """Add an expense and check budget."""
        self.spent_this_month += cost
        remaining = self.monthly_budget - self.spent_this_month

        if remaining < 0:
            print(f"⚠️ BUDGET EXCEEDED! Over by ${abs(remaining):.2f}")
        elif remaining < 1.0:
            print(f"⚠️ Low budget: ${remaining:.2f} remaining")
        else:
            print(f"✅ Budget OK: ${remaining:.2f} remaining")

    def get_status(self):
        """Get budget status."""
        remaining = self.monthly_budget - self.spent_this_month
        return f"💰 Budget: ${remaining:.2f} of ${self.monthly_budget:.2f} remaining"
    
## Q&A
@tool
def mini_qa(channel_id: str, question: str) -> str:
    """Answer intelligent, project-specific questions or perform regenerations like title/description/thumbnail based on the YouTube channel data."""

    # Extract additional context from question if provided
    video_idea_context = None
    title_style_context = None
    
    # Check if question contains context information
    question_lines = question.split('\n')
    for line in question_lines:
        if line.startswith('video_idea:'):
            video_idea_context = line.replace('video_idea:', '').strip()
        elif line.startswith('title_style:'):
            title_style_context = line.replace('title_style:', '').strip()
    
    # Clean the actual question
    actual_question = question_lines[0] if question_lines else question

    api_key = os.getenv("YOUTUBE_API_KEY")
    youtube = build("youtube", "v3", developerKey=api_key)

    # Step 1: Fetch channel and playlist
    try:
        channel_resp = youtube.channels().list(part="snippet,contentDetails,statistics", id=channel_id).execute()
        if not channel_resp.get("items"):
            return "❌ Channel not found."

        channel_name = channel_resp["items"][0]["snippet"]["title"]
        video_count = channel_resp["items"][0]["statistics"]["videoCount"]
        uploads_playlist = channel_resp["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    except Exception as e:
        return f"❌ Error fetching channel data: {e}"

    # Step 2: Fetch up to 30 videos
    videos = []
    next_page_token = None
    try:
        while len(videos) < 30:
            playlist_resp = youtube.playlistItems().list(
                part="snippet", playlistId=uploads_playlist, maxResults=30, pageToken=next_page_token
            ).execute()

            for item in playlist_resp["items"]:
                videos.append({
                    "video_id": item["snippet"]["resourceId"]["videoId"],
                    "title": item["snippet"]["title"],
                    "published_at": item["snippet"]["publishedAt"],
                    "description": item["snippet"].get("description", "")[:200]
                })

            next_page_token = playlist_resp.get("nextPageToken")
            if not next_page_token:
                break

        # Step 3: Get video stats
        video_ids = [v["video_id"] for v in videos]
        stats_resp = youtube.videos().list(part="statistics", id=",".join(video_ids)).execute()

        for idx, item in enumerate(stats_resp.get("items", [])):
            stats = item.get("statistics", {})
            videos[idx]["views"] = int(stats.get("viewCount", 0))
            videos[idx]["likes"] = int(stats.get("likeCount", 0))
            videos[idx]["comments"] = int(stats.get("commentCount", 0))

        df = pd.DataFrame(videos)
        df["published_at"] = pd.to_datetime(df["published_at"])
        df["hour"] = df["published_at"].dt.hour
        df["day"] = df["published_at"].dt.day_name()

        # ===============================
        # 🔍 Q&A LOGIC
        # ===============================
        q = question.lower()

        # ✅ Regenerate Title
        if "regenerate title" in q or "new title" in q or "different title" in q:
            video_idea_to_use = video_idea_context or "general content"
            title_style_to_use = title_style_context or "Trendy"
            return generate_title(video_idea_to_use, title_style_to_use)

        # ✅ Regenerate description  
        if "regenerate description" in q or "new description" in q or "different description" in q:
            video_idea_to_use = video_idea_context or "general content" 
            title_style_to_use = title_style_context or "Trendy"
            return generate_description(video_idea_to_use, title_style_to_use)

        # ✅ Regenerate Thumbnail
        if "regenerate thumbnail" in q or "new thumbnail" in q or "different thumbnail" in q:
            video_idea_to_use = video_idea_context or "YouTube content"
            video_title_to_use = video_idea_context or "Engaging YouTube Video"
            return generate_smart_thumbnail_with_cost_tracking(video_idea_to_use, video_title_to_use)

        # ✅ Best performing video
        if "best" in q and "video" in q:
            top = df.sort_values("views", ascending=False).iloc[0]
            return f"🏆 Top video on {channel_name}: **{top['title']}** with {top['views']:,} views."

        # ✅ Best time to post
        if "best" in q and "time" in q:
            grouped = df.groupby(["day", "hour"])["views"].mean().sort_values(ascending=False)
            best = grouped.index[0]
            return f"⏰ Best time to post: **{best[0]} at {best[1]}:00** based on average views."

        # ✅ Total videos
        if "how many" in q and "videos" in q:
            return f"📺 Total videos: {len(df)}"

        # ✅ Overview / summary
        if "summary" in q or "overview" in q:
            avg = df["views"].mean()
            total = df["views"].sum()
            return f"""
📊 **Summary for {channel_name}:**
- 🎞️ Total Videos on Channel: {video_count}
- 📊 Videos Analyzed in This Report: {len(df)}
- 👀 Total Views (Analyzed): {total:,}
- 📈 Avg Views per Video (Analyzed): {avg:,.0f}
            """

        # ❔ Default help
        return """
🤖 I can answer:
- "What's my best performing video?"
- "What's the best time to post?"
- "Give me a summary"
- "How many videos do I have?"
- "Regenerate title"
- "Regenerate description"
- "Regenerate thumbnail"
"""

    except Exception as e:
        return f"❌ Error during Q&A processing: {e}"
    
## Tools Definition
tools = [get_channel_name,
         generate_title,
         generate_description,
         classify_and_visualize_content_interactive,
         video_scheduler_advisor,
         generate_smart_thumbnail_with_cost_tracking,
         mini_qa]

memory = MemorySaver()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
agent = create_react_agent(model, tools, checkpointer=memory)


config = {"configurable": {"thread_id": "yt-agent-trackb-01"}}
## Main Agent Configuration

def run_agent(channel_id: str, video_idea: str, title_style: str):
    config = {"configurable": {"thread_id": "yt-agent-trackb-01"}}

    video_prompt = f"""I have a new video idea: "{video_idea}".
Please:
- First, show me the channel name for channel ID: {channel_id}
- Analyze my channel content and create comprehensive INTERACTIVE visualizations
- Suggest 3 catchy titles in {title_style} style
- Create a short YouTube description
- Recommend the BEST TIME to post this video based on my channel's performance
- Generate an intelligent, attractive thumbnail optimized for my content type
- Q&A section displaying

Use the tools you need, and show all results including the scheduling recommendation."""

    full_output = ""
    thumbnail_url = None

    for step in agent.stream({"messages": [HumanMessage(content=video_prompt)]}, config, stream_mode="values"):

        msg = step["messages"][-1].content
        full_output += msg + "\n"
        match = re.search(r'https:\/\/[^\s]+(?:\.png|\.jpg|\.jpeg)', msg)
        if match:
            thumbnail_url = match.group(0)
    
    return full_output, thumbnail_url



def run_qa(channel_id: str, user_question: str):

    config = {"configurable": {"thread_id": "yt-agent-trackb-01"}}
    qa_prompt = f"""channel_id: {channel_id}
question: {user_question}"""

    full_answer = ""
    for step in agent.stream({"messages": [HumanMessage(content=qa_prompt)]}, config, stream_mode="values"):
        full_answer += step["messages"][-1].content + "\n"
    return full_answer
