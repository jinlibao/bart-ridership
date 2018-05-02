#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = 'all'

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

__author__ = 'Libao Jin'
__date__ = '04/24/2018'
__email__ = 'ljin1@uwyo.edu'
__copyright__ = 'Copyright (c) 2018 Libao Jin'


def bart_preprocess(bart):
    # Convert string to datetime, split datetime into date and time, and get day of week
    bart['DateTime'] = pd.to_datetime(bart.DateTime)
    bart['Date'] = bart['DateTime'].dt.date
    bart['Time'] = bart['DateTime'].dt.time
    bart['DayOfWeek'] = bart['DateTime'].dt.weekday_name
    # bart.drop(columns='DateTime')
    return bart


def stat_preprocess(stat):
    # Split location into longitude and latitude for visualization later on
    loc = stat.Location.str.split(',', expand=True)
    loc = [pd.to_numeric(loc[i]) for i in loc.columns]
    stat['Longitude'], stat['Latitude'] = loc[0], loc[1]
    columns = ['Abbreviation', 'Name', 'Longitude', 'Latitude', 'Description']
    stat = stat[columns]
    return stat


def generate_bart_routes(bart_lines, stat):
    bart_routes = []

    for line in bart_lines:

        abbr = []
        name = []
        fullname = []
        location = []

        for station in line:
            tmp = stat[stat['Name'].str.contains(station)]
            abbr.append(tmp.iloc[0]['Abbreviation'])
            name.append(station)
            fullname.append(tmp.iloc[0]['Name'])
            location.append([tmp.iloc[0]['Longitude'],
                             tmp.iloc[0]['Latitude']])

        bart_routes.append({
            'abbr': abbr,
            'name': name,
            'fullname': fullname,
            'location': location
        })
    return bart_routes


def visualize_bart_routes(bart_routes, filename='bart_routes.pdf'):
    color = list('krbcy')
    alpha = [0.9, 0.8, 0.7, 0.6, 0.5]
    x_min, x_max, y_min, y_max = -122.62, -121.75, 37.48, 38.04

    with PdfPages(filename) as pdf:
        plt.style.use('default')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        for i in range(len(bart_routes)):
            fig = plt.figure(figsize=(16, 12))
            loc = np.array(bart_routes[i]['location'])
            # x, y = loc[:, 0], loc[:, 1]
            # plt.plot(x, y, '-o', c=color[i], ms=6, lw=4, alpha=0.5)
            plt.plot(
                loc[:, 0],
                loc[:, 1],
                '-o',
                c=color[i],
                mfc='w',
                ms=8,
                lw=9,
                alpha=0.75
            )
            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            title = '{} - {}'.format(bart_routes[i]['fullname'][0],
                                     bart_routes[i]['fullname'][-1])
            plt.title(title)
            plt.show(block=False)
            pdf.savefig(fig)

        fig = plt.figure(figsize=(16, 12))
        for i in range(len(bart_routes)):
            loc = np.array(bart_routes[i]['location'])
            label = 'Line {:d}: {} - {}'.format(i + 1,
                                                bart_routes[i]['abbr'][0],
                                                bart_routes[i]['abbr'][-1])
            # x, y = loc[:, 0], loc[:, 1]
            # plt.plot(x, y, '-o', c=color[i], ms=6, lw=4, alpha=0.5)
            plt.plot(
                loc[:, 0],
                loc[:, 1],
                '-o',
                c=color[i],
                mfc='w',
                ms=8,
                lw=9,
                alpha=alpha[i],
                label=label
            )
        plt.legend()
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('BART Lines')
        plt.show(block=False)
        # plt.axis('equal')
        pdf.savefig(fig)


def visualize_throughput(bart_routes, bart_aggregate, column_name, filename='bart_throughput.pdf'):
    with PdfPages(filename) as pdf:
        color = list('krbcy')
        bart_aggregate = bart_aggregate.set_index('Station')
        x, y = bart_aggregate['Longitude'], bart_aggregate['Latitude']
        n = len(bart_aggregate['Longitude'])
        cmap = mpl.cm.cool
        cs = getattr(bart_aggregate, column_name)
        x_min, x_max, y_min, y_max = -122.62, -121.75, 37.48, 38.04
        fig = plt.figure(figsize=(16, 12))
        plt.style.use('default')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        for i in range(len(bart_routes)):
            loc = np.array(bart_routes[i]['location'])
            label = 'Line {:d}: {} - {}'.format(i + 1,
                                                bart_routes[i]['abbr'][0],
                                                bart_routes[i]['abbr'][-1])
            # x, y = loc[:, 0], loc[:, 1]
            # plt.plot(x, y, '-o', c=color[i], ms=6, lw=4, alpha=0.5)
            plt.plot(
                loc[:, 0],
                loc[:, 1],
                '-',
                c=color[i],
                # mfc='w',
                # ms=8,
                # lw=6,
                alpha=0.5,
                label=label,
                zorder=1
            )
        plt.legend()

        plt.scatter(
            x,
            y,
            c=cs,
            s=cs,
            marker='o',
            edgecolors='k',
            cmap=cmap,
            alpha=1,
            zorder=2
        )

        plt.colorbar()
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.axis('equal')
        plt.title('BART Lines ' + cs.name)
        plt.show(block=False)
        pdf.savefig(fig)

        fig = plt.figure(figsize=(16, 12))
        plt.style.use('ggplot')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        bart_aggregate.sort_values(column_name, inplace=True, ascending=False)
        cs = getattr(bart_aggregate, column_name)
        colors = cmap(np.linspace(1, 0, n))
        cs.plot(kind='bar', color=colors, alpha=0.75)
        # print(list(cs.index))
        plt.ylabel('Average Throughput / hour')
        plt.title('Throughput vs. Station')
        plt.show(block=False)
        pdf.savefig(fig)


def bart_aggregate_throughput(bart, filename):
    number_of_days = len(bart['DateTime'].dt.date.unique())
    number_of_hours = len(bart['DateTime'].dt.time.unique())
    bart_grouped = bart['Throughput'].groupby(bart['Origin']).sum().to_frame()
    bart_grouped['Destination'] = bart['Throughput'].groupby(bart['Destination']).sum()
    bart_grouped.index.names = ['Station']
    bart_grouped.columns = ['Throughput Origin', 'Throughput Destination']
    bart_grouped['Throughput All'] = bart_grouped['Throughput Origin'] + bart_grouped['Throughput Destination']
    bart_grouped.set_index(stat['Abbreviation'])
    bart_grouped.reset_index(level=0, inplace=True)
    bart_grouped['Longitude'] = stat['Longitude']
    bart_grouped['Latitude'] = stat['Latitude']
    bart_grouped[['Throughput Origin', 'Throughput Destination', 'Throughput All']] = \
        bart_grouped[['Throughput Origin', 'Throughput Destination', 'Throughput All']] / (number_of_days * number_of_hours)
    visualize_throughput(bart_routes, bart_grouped, 'Throughput Origin', filename)
    # visualize_throughput(bart_routes, bart_grouped, 'Throughput Destination', filename)
    # visualize_throughput(bart_routes, bart_grouped, 'Throughput All', filename)


def visualize_bart(bart, origins, stat, class_type, group_by, plot_option, filename='bart_overview.pdf'):

    plt.style.use('ggplot')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    with PdfPages(filename) as pdf:
        for stops in origins:
            if len(stops) == 1:
                data = bart[bart['Origin'] == stops[0]]
            elif len(stops) == 2:
                origin, dest = stops
                data = bart[(bart['Origin'] == origin) & (bart['Destination'] == dest)]
            # Plot the throughput with respect to time (hour) each week/month
            if plot_option == 1 or group_by == 'hour':
                k = len(getattr(data['DateTime'].dt, class_type[0]).unique())
                # n = 2                                 # number of columns
                # m = math.ceil(k / n)  # number of rows
                # plt.figure(figsize=(8 * n, 6 * m))
                for i in range(k):
                    fig = plt.figure(figsize=(8, 6))
                    if class_type[0] == 'weekday':
                        j = i
                    else:
                        j = i + 1
                    grouped = data[getattr(data['DateTime'].dt, class_type[0]).values == j].groupby(
                        getattr(data['DateTime'].dt, group_by)).sum()
                    grouped.sort_index(inplace=True)
                    plt.plot(grouped['Throughput'])
                    if len(stops) == 1:
                        tmp = stat[stat['Abbreviation'] == data.iloc[0]['Origin']]
                        title = '{}: {}'.format(class_type[1][i],
                                                tmp.iloc[0]['Name'])
                    else:
                        tmp_1 = stat[stat['Abbreviation'] == data.iloc[0]['Origin']]
                        tmp_2 = stat[stat['Abbreviation'] == data.iloc[0]['Destination']]
                        title = '{}: {} - {}'.format(class_type[1][i],
                                                     tmp_1.iloc[0]['Name'],
                                                     tmp_2.iloc[0]['Name'])
                    plt.title(title)
                    plt.xlabel(group_by.title())
                    ax = plt.gca()
                    # fig = plt.gcf()
                    if group_by == 'date':
                        xfmt = mpl.dates.DateFormatter('%Y-%m-%d')
                        ax.xaxis.set_major_formatter(xfmt)
                        # plt.xticks(rotation=90)
                        fig.autofmt_xdate()
                    plt.ylabel('Total Throughput')
                    plt.ylim([-10000, 780000])
                    # plt.xticks(grouped.index, list(np.arange(24)))
                    plt.show(block=False)
                    pdf.savefig(fig)

            # Plot the throughput with respect to date
            elif plot_option == 2:
                grouped = data.groupby(getattr(data['DateTime'].dt, group_by)).sum()
                grouped.sort_index(inplace=True)
                fig = plt.figure(figsize=(8, 6))
                plt.plot(grouped['Throughput'])
                if len(stops) == 1:
                    tmp = stat[stat['Abbreviation'] == data.iloc[0]['Origin']]
                    title = '{}'.format(tmp.iloc[0]['Name'])
                    # title = class_type[1][i] + ': {}'.format(data.iloc[0]['Origin'])
                else:
                    tmp_1 = stat[stat['Abbreviation'] == data.iloc[0]['Origin']]
                    tmp_2 = stat[stat['Abbreviation'] == data.iloc[0]['Destination']]
                    title = '{} - {}'.format(tmp_1.iloc[0]['Name'],
                                             tmp_2.iloc[0]['Name'])
                    # title = ': {} - {}'.format(data.iloc[0]['Origin'], data.iloc[0]['Destination'])
                # plt.legend()
                plt.xlabel(group_by.title())
                plt.ylabel('Throughput')
                plt.xticks(rotation=90)
                ax = plt.gca()
                # fig = plt.gcf()
                xfmt = mpl.dates.DateFormatter('%Y-%m-%d')
                ax.xaxis.set_major_formatter(xfmt)
                # plt.xticks(rotation=90)
                fig.autofmt_xdate()
                plt.title(title)
                plt.ylim([0, 65000])
                plt.show(block=False)
                pdf.savefig(fig)


if __name__ == '__main__':

    dest_folder = './output'
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # Load data
    # Data obtained from https://www.kaggle.com/saulfuh/bart-ridership
    date_hour_2016 = '../data/bart-ridership/date-hour-soo-dest-2016.csv'
    date_hour_2017 = '../data/bart-ridership/date-hour-soo-dest-2017.csv'
    stat_info = '../data/bart-ridership/station_info.csv'
    # bart_16 = pd.read_csv(date_hour_2016)
    # bart_17 = pd.read_csv(date_hour_2017)
    # stat = pd.read_csv(stat_info)

    # Data preprocessing
    bart_16 = bart_preprocess(pd.read_csv(date_hour_2016))
    bart_17 = bart_preprocess(pd.read_csv(date_hour_2017))
    bart = pd.concat([bart_16, bart_17], ignore_index=True)
    stat = stat_preprocess(pd.read_csv(stat_info))

    # Visualize the routes according to the BART official website

    line_1 = [
        'Richmond',
        'El Cerrito del Norte',
        'El Cerrito Plaza',
        'North Berkeley',
        'Downtown Berkeley',
        'Ashby',
        'West Oakland',
        'Embarcadero',
        'Montgomery St.',
        'Powell St.',
        'Civic Center/UN Plaza',
        'Daly City',
        'Colma',
        'South San Francisco',
        'San Bruno',
        'Millbrae'
    ]

    line_2 = [
        'Richmond',
        'El Cerrito del Norte',
        'El Cerrito Plaza',
        'North Berkeley',
        'Downtown Berkeley',
        'Ashby',
        'MacArthur',
        '19th St. Oakland',
        '12th St. Oakland City Center',
        'Lake Merritt',
        'Fruitvale',
        'Coliseum/Oakland Airport',
        'San Leandro',
        'Bay Fair',
        'Hayward',
        'South Hayward',
        'Union City',
        'Fremont'
    ]

    line_3 = [
        'Pittsburg/Bay Point',
        'North Concord/Martinez',
        'Concord',
        'Walnut Creek',
        'Lafayette',
        'Orinda',
        'Rockridge',
        'MacArthur',
        '19th St. Oakland',
        '12th St. Oakland City Center',
        'West Oakland',
        'Embarcadero',
        'Montgomery St.',
        'Powell St.',
        'Civic Center/UN Plaza',
        '16th St. Mission',
        '24th St. Mission',
        'Glen Park',
        'Balboa Park',
        'Daly City',
        'Colma',
        'South San Francisco',
        'San Bruno',
        "San Francisco Int'l Airport",
        'Millbrae'
    ]

    line_4 = [
        'Dublin/Pleasanton',
        'West Dublin/Pleasanton',
        'Castro Valley',
        'Bay Fair',
        'San Leandro',
        'Coliseum/Oakland Airport',
        'Fruitvale',
        'Lake Merritt',
        'West Oakland',
        'Embarcadero',
        'Montgomery St.',
        'Powell St.',
        'Civic Center/UN Plaza',
        '16th St. Mission',
        '24th St. Mission',
        'Glen Park',
        'Balboa Park',
        'Daly City'
    ]

    line_5 = [
        'Warm Springs/South Fremont',
        'Fremont',
        'Union City',
        'South Hayward',
        'Hayward',
        'Bay Fair',
        'San Leandro',
        'Coliseum/Oakland Airport',
        'Fruitvale',
        'Lake Merritt',
        'West Oakland',
        'Embarcadero',
        'Montgomery St.',
        'Powell St.',
        'Civic Center/UN Plaza',
        '16th St. Mission',
        '24th St. Mission',
        'Glen Park',
        'Balboa Park',
        'Daly City'
    ]

    bart_lines = [
        line_1,
        line_2,
        line_3,
        line_4,
        line_5
    ]

    bart_routes = generate_bart_routes(bart_lines, stat)
    visualize_bart_routes(bart_routes, '{}/bart_routes.pdf'.format(dest_folder))

    # bart_aggregate_throughput(bart_16)
    # bart_aggregate_throughput(bart_17)
    bart_aggregate_throughput(bart, '{}/bart_throughput.pdf'.format(dest_folder))

    by_month = [
        'month',
        ['January', 'Febrary', 'March', 'April', 'May', 'June', 'July',
         'August', 'September', 'October', 'November', 'December']
    ]

    by_weekday = [
        'weekday',
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
         'Friday', 'Saturday', 'Sunday']
    ]

    origins_names = [
        'MONT', 'EMBR', 'POWL', 'CIVC', '24TH', '16TH', '12TH',
        'DBRK', '19TH', 'BALB', 'DALY', 'MCAR', 'FRMT', 'DELN',
        'FTVL', 'DUBL', 'GLEN', 'WOAK', 'SFIA', 'COLS', 'PHIL',
        'LAKE', 'MLBR', 'WCRK', 'PITT', 'CONC', 'SANL', 'BAYF',
        'ROCK', 'ASHB', 'PLZA', 'HAYW', 'UCTY', 'NBRK', 'RICH',
        'COLM', 'SBRN', 'SSAN', 'LAFY', 'WDUB', 'SHAY', 'CAST',
        'ORIN', 'NCON', 'OAKL'
    ]

    origins = [[i] for i in origins_names]
    visualize_bart(bart, origins, stat, by_month, 'date', 2, '{}/bart_overview_1.pdf'.format(dest_folder))
    visualize_bart(bart, origins, stat, by_weekday, 'hour', 2, '{}/bart_overview_2.pdf'.format(dest_folder))
    # origins2 = [[origins_names[i], origins_names[i + 1]] for i in range(len(origins_names) - 1)]
    # visualize_bart(bart, origins2, stat, by_month, 'hour', 1, '{}/bart_overview_3.pdf'.format(dest_folder))
