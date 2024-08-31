'''
Descriptive analysis of the dataset (message_frequency, etc.)
'''


#import libraries
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DayLocator, WeekdayLocator, DateFormatter
import plotly.express as px


#define helper functions
def format_df(embassy_df):

    '''
    apply literal_eval to selected columns of the dataframe
    add channel_username, channel_id and channel_title column for ease of processing
    convert the 'date' column to datetime, set as index, sort to allow slicing
    '''
    embassy_df['_chat'] = embassy_df['_chat'].apply(literal_eval)
    embassy_df['channel_id'] = embassy_df['_chat'].apply(lambda x: x['id'])
    embassy_df['channel_title'] = embassy_df['_chat'].apply(lambda x: x['title'])
    embassy_df['channel_username'] = embassy_df['_chat'].apply(lambda x: x['username'])
    embassy_df['date'] = pd.to_datetime(embassy_df['date']) 
    embassy_df = embassy_df.set_index('date')
    embassy_df = embassy_df.sort_index() 

    return embassy_df

def get_forwarded_msg_df(embassy_df):
    '''
    returns a dataframe containing only forwarded messages
    '''

    forwarded_msg_df = embassy_df[~embassy_df['fwd_from'].isnull()]

    return forwarded_msg_df

def get_original_msg_df(embassy_df):
    '''
    returns a dataframe containing only original messages 
    '''

    original_msg_df = embassy_df[embassy_df['fwd_from'].isnull()]

    return original_msg_df

def get_txt_msg_df(embassy_df):
    '''
    returns a dataframe containing only messages with text
    '''

    text_msg_df = embassy_df[~embassy_df['message'].isnull()]

    return text_msg_df

def get_no_txt_msg_df(embassy_df):
    '''
    returns dataframe containing only messages without text
    '''

    no_txt_msg_df = embassy_df[embassy_df['message'].isnull()]

    return no_txt_msg_df

def count_rows(df):
    '''
    return number of rows in a dataframe
    '''
    return len(df.index)


def plot_activity_gantt(embassy_channel_df, output_path, plot_title):
    '''
    Make a Gantt chart that shows the channel creation data and the period for which it has remained active
    '''
    
    #create dataframe
    gantt_df = pd.DataFrame()

    #initiate lists for data
    channel_names = []
    first_messages = []
    last_messages = []
    channel_creation_dates = []

    #get channel name, channel creation data, channel first message, channel last message
    for channel_name, channel_df in embassy_df.groupby(['channel_username']):
        channel_name = channel_name[0]
        first_message = channel_df.index.min()
        last_message = channel_df.index.max()
        channel_creation_date = pd.to_datetime(channel_df['_chat'][0]['date'], unit = "s")

        channel_names.append(channel_name)
        first_messages.append(first_message)
        last_messages.append(last_message)
        channel_creation_dates.append(channel_creation_date)

    #complete the dataframe
    gantt_df['channel_name'] = channel_names
    gantt_df['creation_date'] = channel_creation_dates
    gantt_df['first_message'] = first_messages
    gantt_df['last_message'] = last_messages

    #sort by creation data
    gantt_df = gantt_df.sort_values(by=['creation_date'])

    #make the gantt chart
    #convert date columns to datetime
    gantt_df['creation_date'] = pd.to_datetime(gantt_df['creation_date'])
    gantt_df['first_message'] = pd.to_datetime(gantt_df['first_message'])
    gantt_df['last_message'] = pd.to_datetime(gantt_df['last_message'])

    #create the plot
    fig, ax = plt.subplots(figsize=(10, 25))

    #define y positions for each channel
    y_pos = range(len(gantt_df))

    #plot the creation dates
    ax.scatter(gantt_df['creation_date'], y_pos, color='blue', label='channel creation date', zorder=3)

    #plot the periods between first and last messages
    for i, (start, end) in enumerate(zip(gantt_df['first_message'], gantt_df['last_message'])):
        ax.plot([start, end], [i, i], color='grey', linewidth= 5, zorder=2, solid_capstyle='butt')

    #customize the y-axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels(gantt_df['channel_name'])
    ax.invert_yaxis() 

    #format x-axis to show dates
    ax.xaxis_date()
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    plt.xticks(rotation=45)

    #format secondary axis on top of graph
    ax_top = ax.twiny()
    ax_top.xaxis_date()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.xaxis.set_major_locator(YearLocator())
    ax_top.xaxis.set_major_formatter(DateFormatter('%Y'))

    #add labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Channel')
    ax.set_title(plot_title)

    #add a legend
    ax.legend()

    #add vertical gridlines
    ax.grid(axis='x', linestyle='--', linewidth=0.5)

    #add horizontal lines for legibility
    for y in y_pos:
        ax.axhline(y=y, color='gray', linestyle='dotted', linewidth=0.5, zorder=1)

    #add translucent blue square for the scraping window
    scraping_start_date = pd.to_datetime('2020-02-01 00:00:00')
    scraping_end_date = pd.to_datetime('2024-03-01 00:00:00')
    ax.axvspan(scraping_start_date, scraping_end_date, color='lightblue', alpha=0.3, zorder=0)

    #add red line and label marking invasion
    plt.axvline(x=pd.Timestamp('2022-02-24 00:00:00'), color = "red")
    ylim = ax.get_ylim() 
    plt.text(pd.Timestamp('2022-02-24 00:00:00'), ylim[0] + 0.97 * (ylim[1] - ylim[0]), '  2022-02-24', color='red', verticalalignment='center')

    #layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi = 600)


def plot_zoomed_activity_gantt(embassy_channel_df, output_path, plot_title, start_date, end_date):
    '''
    Make a Gantt chart that shows the channel creation data and the period for which it has remained active
    '''
    
    #create dataframe
    gantt_df = pd.DataFrame()

    #initiate lists for data
    channel_names = []
    first_messages = []
    last_messages = []
    channel_creation_dates = []

    #get channel name, channel creation data, channel first message, channel last message
    for channel_name, channel_df in embassy_df.groupby(['channel_username']):
        channel_name = channel_name[0]
        first_message = channel_df.index.min()
        last_message = channel_df.index.max()
        channel_creation_date = pd.to_datetime(channel_df['_chat'][0]['date'], unit = "s")

        channel_names.append(channel_name)
        first_messages.append(first_message)
        last_messages.append(last_message)
        channel_creation_dates.append(channel_creation_date)

    #complete the dataframe
    gantt_df['channel_name'] = channel_names
    gantt_df['creation_date'] = channel_creation_dates
    gantt_df['first_message'] = first_messages
    gantt_df['last_message'] = last_messages

    #sort by creation data
    gantt_df = gantt_df.sort_values(by=['creation_date'])

    #make the gantt chart
    #convert date columns to datetime
    gantt_df['creation_date'] = pd.to_datetime(gantt_df['creation_date'])
    gantt_df['first_message'] = pd.to_datetime(gantt_df['first_message'])
    gantt_df['last_message'] = pd.to_datetime(gantt_df['last_message'])

    #create the plot
    fig, ax = plt.subplots(figsize=(10, 25))

    #define y positions for each channel
    y_pos = range(len(gantt_df))

    #plot the creation dates
    ax.scatter(gantt_df['creation_date'], y_pos, color='blue', label='channel creation date', zorder=3)

    #plot the periods between first and last messages
    for i, (start, end) in enumerate(zip(gantt_df['first_message'], gantt_df['last_message'])):
        ax.plot([start, end], [i, i], color='grey', linewidth= 5, zorder=2, solid_capstyle='butt')

    #limit the scope of the graph to a specific timeframe
    ax.set_xlim([pd.Timestamp(start_date), pd.Timestamp(end_date)])

    #customize the y-axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels(gantt_df['channel_name'])
    ax.invert_yaxis() 

    #format x-axis to show dates
    ax.xaxis_date()
    ax.xaxis.set_major_locator(WeekdayLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%b %d'))
    plt.xticks(rotation=45)

    #format secondary axis on top of graph
    ax_top = ax.twiny()
    ax_top.xaxis_date()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.xaxis.set_major_locator(WeekdayLocator())
    ax_top.xaxis.set_major_formatter(DateFormatter('%b %d'))
    plt.xticks(rotation=45)

    #add labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Channel')
    ax.set_title(plot_title)

    #add a legend
    ax.legend()

    #add vertical gridlines
    ax.grid(axis='x', linestyle='--', linewidth=0.5)

    #add horizontal lines for legibility
    for y in y_pos:
        ax.axhline(y=y, color='gray', linestyle='dotted', linewidth=0.5, zorder=1)

    #add translucent blue square for the scraping window
    scraping_start_date = pd.to_datetime('2020-02-01 00:00:00')
    scraping_end_date = pd.to_datetime('2024-03-01 00:00:00')
    ax.axvspan(scraping_start_date, scraping_end_date, color='lightblue', alpha=0.3, zorder=0)

    #add red line and label marking invasion
    plt.axvline(x=pd.Timestamp('2022-02-24 00:00:00'), color = "red")
    ylim = ax.get_ylim() 
    plt.text(pd.Timestamp('2022-02-24 00:00:00'), ylim[0] + 0.97 * (ylim[1] - ylim[0]), '  2022-02-24', color='red', verticalalignment='center')

    #layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi = 600)


def plot_message_frequency(embassy_df, output_path):
    '''
    plot message frequency over time, mark significant events
    '''

    #cross-temporal analysis of message types by hour in the days leading up to the invasion
    forwarded_messages = get_forwarded_msg_df(embassy_df)
    original_messages = get_original_msg_df(embassy_df)

    forwarded_row_counts = forwarded_messages.resample('D').size()
    original_row_counts = original_messages.resample('D').size()

    #ensure that there is a common index
    common_index = forwarded_row_counts.index.union(original_row_counts.index)
    forwarded_row_counts = forwarded_row_counts.reindex(common_index, fill_value=0)
    original_row_counts = original_row_counts.reindex(common_index, fill_value=0)

    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)

    forwarded_bars = plt.bar(forwarded_row_counts.index, forwarded_row_counts.values, label='Forwarded messages')
    original_bars = plt.bar(original_row_counts.index, original_row_counts.values, bottom=forwarded_row_counts.values, label='Original messages')

    plt.title('Message per day (all channels)')
    plt.xlabel('Time')
    plt.ylabel('Number of messages')
    plt.legend()
    plt.ylim(0, 1100) #look at daily messages
    plt.xlim(pd.Timestamp('2022-01-01 00:00:00'), pd.Timestamp('2022-04-01 00:00:00'))
    plt.axvline(x=pd.Timestamp('2022-02-24 00:00:00'), color = "red") #add a vertical line indicating the time of the Russian invasion of Ukraine
    plt.text(pd.Timestamp('2022-02-24 00:00:00'), 1100 * 0.8, '  2022-02-24', color='red', verticalalignment='center')
    plt.grid(axis='y', linestyle='--')
    plt.savefig(output_path, dpi = 600)


def get_embassy_channel_list(embassy_df, file):
    ''' 
    write list of channels to txt file
    '''
    embassy_channel_list = list(set(list(embassy_df['channel_username'])))
    with open(file, 'w') as outfile:
        outfile.write('\n'.join(str(i) for i in embassy_channel_list))


def plot_treemap(embassy_df, output_path):
    '''
    draw a treemap diagram of the dataset, giving a proportional view of message types
    '''

    #identify message types
    print('identify message types')
    message_types_df = pd.DataFrame()
    message_types_df['message_types'] = ['forwarded' if pd.notna(msg) and msg != '' else 'original' for msg in embassy_df['fwd_from']]
    message_types_df['has_text'] = ['text' if pd.notna(msg) and msg != '' else 'no text' for msg in embassy_df['message']]
   
    #clean up language labels 
    clean_language_labels = []

    for language_label in list(embassy_df['message_language']):
        if type(language_label) == str:
            language_label = ', '.join(literal_eval(language_label))
        else:
            language_label = None
        clean_language_labels.append(language_label)

    message_types_df['language'] = clean_language_labels

    #set language to none if there is no text
    message_types_df.loc[message_types_df['has_text'] == 'no text', 'language'] = 'no language'

    #create a frequency column
    message_types_df['count'] = 1

    #aggregate the counts
    aggregated_df = message_types_df.groupby(['message_types', 'has_text', 'language']).sum().reset_index()

    #plot treemap
    print('plot treemap')
    fig = px.treemap(
        aggregated_df,
        path=['message_types', 'has_text', 'language'],
        values='count',
        #title='composition of dataset by message type' #title to be added in paper
    )

    #show absolute numbers
    fig.update_traces(textinfo='label+value')

    #clean up the margins of the figure
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

    print('save figure')
    fig.write_image(output_path, engine='kaleido', scale= 3)


def plot_views(embassy_df, output_path):
    '''
    Plot message views (aggregated by timestamp)
    '''

    print('resample and count views') #to do: split between fowarded and regular messages?
    daily_views = embassy_df['views'].resample('W').sum()
    daily_views.plot(kind='bar', figsize=(10, 6))
    plt.xlabel('Date')
    plt.ylabel('View Count')
    plt.title('Weekly View Counts')

    print('save figure')
    plt.savefig(output_path)


def create_dataset_table(embassy_channel_df, output_path):
    '''
    Create a table with an overview of the channels, their message counts, creation date, and dates active
    To be included as supplementary materials with the paper
    '''
    
    #create dataframe
    overview_df = pd.DataFrame()

    #initiate lists for data
    channel_names = []
    first_messages = []
    last_messages = []
    channel_creation_dates = []
    channel_message_counts = []

    #get channel name, channel creation data, channel first message, channel last message
    for channel_name, channel_df in embassy_df.groupby(['channel_username']):

        channel_name = channel_name[0]
        first_message = channel_df.index.min()
        last_message = channel_df.index.max()
        channel_creation_date = pd.to_datetime(channel_df['_chat'][0]['date'], unit = "s")
        channel_message_count = len(channel_df.index)


        channel_names.append(channel_name)
        first_messages.append(first_message)
        last_messages.append(last_message)
        channel_creation_dates.append(channel_creation_date)
        channel_message_counts.append(channel_message_count)

    #we manually append the information for the embassy in Gabon
    channel_names.append('ambrusga')
    first_messages.append(None)
    last_messages.append(None)
    channel_creation_dates.append('2022-03-21')
    channel_message_counts.append(0)

    #complete the dataframe
    overview_df['channel_name'] = channel_names
    overview_df['creation_date'] = channel_creation_dates
    overview_df['first_collected_message'] = first_messages
    overview_df['last_collected_message'] = last_messages
    overview_df['message_count'] = channel_message_counts

    #sort the dataframe alphabetically
    overview_df = overview_df.sort_values('channel_name')

    #store the dataframe
    overview_df.to_csv(output_path, index = False)


def plot_first_message_gantt(embassy_df, output_path, plot_title, start_date, end_date):
    '''
    Create a zoomed-in Gantt chart that shows the time between channel creation and first message for a given timeframe
    '''

    print('create first messages df')

    first_message_df = pd.DataFrame()
    channel_names = []
    first_message_dates = []
    first_message_texts = []
    channel_creation_dates = []

    for channel_name, channel_df in embassy_df.groupby(['channel_username']):
        channel_names.append(channel_name[0])
        first_message_dates.append(channel_df.index.min())
        first_message_texts.append(channel_df['message'][0])
        channel_creation_dates.append(pd.to_datetime(channel_df['_chat'][0]['date'], unit = "s"))

    first_message_df['channel_username'] = channel_names
    first_message_df['first_message_date'] = first_message_dates
    first_message_df['first_message_text'] = first_message_texts
    first_message_df['channel_creation_date'] = channel_creation_dates

    #filter for creation dates in February 2022
    first_message_df = first_message_df[(first_message_df['channel_creation_date'] >= start_date) & (first_message_df['channel_creation_date'] < end_date)]

    #sort by creation data
    first_message_df = first_message_df.sort_values(by=['channel_creation_date'])

    #create the plot
    fig, ax = plt.subplots(figsize=(10, 10))

    #define y positions for each channel
    y_pos = range(len(first_message_df))

    #plot the creation dates
    ax.scatter(first_message_df['channel_creation_date'], y_pos, color='blue', label='channel creation date', zorder=3)

    #plot the first message dates
    ax.scatter(first_message_df['first_message_date'], y_pos, color='red', label='first message date', zorder=3)

    #plot the periods between creation date and first message
    for i, (start, end) in enumerate(zip(first_message_df['channel_creation_date'], first_message_df['first_message_date'])):
        ax.plot([start, end], [i, i], color='grey', linewidth= 5, zorder=2, solid_capstyle='butt') 

    #set x-axis limit
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    ax.set_xlim([start_date, end_date])

    #customize the y-axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels(first_message_df['channel_username'])
    ax.invert_yaxis() 

    # Format x-axis to show dates at the day level
    ax.xaxis_date()
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    #add labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Channel')
    ax.set_title(plot_title)

    #add a legend
    ax.legend()

    #add vertical gridlines
    ax.grid(axis='x', linestyle='--', linewidth=0.5)

    #add horizontal lines for legibility
    for y in y_pos:
        ax.axhline(y=y, color='gray', linestyle='dotted', linewidth=0.5, zorder=1)

    #add red line and label marking invasion
    plt.axvline(x=pd.Timestamp('2022-02-24 00:00:00'), color = "red")
    ylim = ax.get_ylim() 
    plt.text(pd.Timestamp('2022-02-24 00:00:00'), ylim[0] + 0.97 * (ylim[1] - ylim[0]), '  2022-02-24', color='red', verticalalignment='center')

    #layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi = 600)

  
if __name__ == "__main__":

    #specify path to data
    csv_sample_file = "/home/tom/Documents/data/geopolitics_of_propaganda/4cat_data_sample.csv"

    #load embassies data
    print('load data')
    embassy_df = pd.read_csv(csv_sample_file)

    #format data 
    print('format data')
    embassy_df = format_df(embassy_df)

    #produce a list of all embassy channels in the dataset, write to file
    print('get embassy channel lists')
    get_embassy_channel_list(embassy_df, 'outputs/lists/channel_usernames.txt')

    #produce a csv with a description of the full dataset
    print('get dataset overview')
    create_dataset_table(embassy_df, 'outputs/lists/dataset_overview.csv')

    #plot dataset composition as treemap
    print('plot treemap')
    plot_treemap(embassy_df,'outputs/figures/fig1_treemap.png')  
    
    # #produce graph of aggregated views over time
    # print('plot views')
    # plot_views(embassy_df, 'outputs/figures/message_views.png')

    #plot channel creation dates and activity (Gantt chart)
    print('plot dataset activity gantt')
    plot_activity_gantt(embassy_df, 'outputs/figures/fig2_activity_gantt.png', 'Gantt chart of channel creation date and activity')

    print('plot zoomed in activity gantt')
    plot_zoomed_activity_gantt(embassy_df, 'outputs/figures/fig3_zoomed_activity_gantt.png', 'zoomed-in Gantt chart of channel creation date and activity', '2022-01-01', '2022-04-01')

    # print('plot first messages gantt')
    # plot_first_message_gantt(embassy_df, 'outputs/figures/fig3_first_message_gantt.png', 'time between channel creation and first post', '2022-02-20', '2022-03-01')

    #plot message frequency over time
    print('plot message frequency over time')
    plot_message_frequency(embassy_df, 'outputs/figures/fig4_message_timeseries.png')
