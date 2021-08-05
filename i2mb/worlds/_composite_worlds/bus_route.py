import tabula
import networkx as nx
import matplotlib.pyplot as plt
import pylab
import numpy as np
from matplotlib.patches import Rectangle, Circle, PathPatch
from matplotlib.path import Path

from i2mb import _assets_dir


def get_busroute(line_number, pdf_name=None):
    """
    :param line_number: number of bus route
    :param pdf_name: exact pdf name, if you want to create a new csv file
    :return: a list made up of two lists, there and return of the route
    """

    if pdf_name != None:
        path_pdf = f"{_assets_dir}/" + pdf_name
        # only if you want a new csv file!!!
        # tabula.convert_into(path_pdf, path_csv, output_format='csv', pages='all')

    path_csv = f"{_assets_dir}\Linie-" + line_number + ".csv"
    file = open(path_csv, 'r')
    bus_route = file.readlines()
    # print(bus_route)

    # test
    # bus_route = ['280,,,,,,\n',
    #             'ESTW - Erlanger Stadtwerke Stadtverkehr GmbH; Äußere Brucker Str. 33; 91052 Erlangen; Tel. 09131 823-4000; www.estw.de; stadtverkehr@estw.de,,,,,,\n',
    #            'è Gültig ab 13.12.2020 Montag-Freitag,,,,,,\n', 'Uhr 5 6 7 - 8 9 10 11,,12,,13 - 17,,18\n',
    #           'Lindnerstraße -Hst 3- 29 59 29 49 09 29 49 09 29 59 29 59 29 59,29,49,09,29,49,09\n',
    #          'Westerwaldweg 31 01 31 51 11 31 51 11 31 01 31 01 31 01,31,51,11,31,51,11\n',
    #         'Joseph-Will-Straße 32 02 32 52 12 32 52 12 32 02 32 02 32 02,32,52,12,32,52,12\n',
    #        'è Gültig ab 13.12.2020 Montag-Freitag,,,,,,', 'Uhr 18 19 20 21,,,,,,',
    #       'Lindnerstraße -Hst 3- 29 59 29 59 29,,,,,,']

    schedule = []
    numbers = "0123456789"
    hours_list = []

    for line in bus_route[:]:
        # deleting unwanted lines
        if "ESTW" in line:
            continue
        if "Gueltig ab" in line:
            continue
        if line_number in line:
            continue
        if "VERKEHRSHINWEIS" in line:
            continue

        # deleting -hst number-
        hst = " -Hst"
        index_hst = line.find(hst)
        if index_hst != -1:
            line = line[:index_hst] + line[index_hst + len(hst) + 3:]

        # get station name
        name_end_index = line.find(" ")
        if line[name_end_index - 1] in numbers:
            continue
        # allowing spaces in Name
        while 'a' <= line[name_end_index + 1] <= 'z' or 'A' <= line[name_end_index + 1] <= 'Z':
            name_end_index = line.find(" ", name_end_index + 1)

        station_name = line[:name_end_index]
        line = line[name_end_index:].replace(",", " ").replace("\n", "").replace("  ", " ")
        # deleting spaces at the end of the minute schedule
        while line[len(line) - 1] == " ":
            line = line[:len(line) - 1]

        # initialize hours_list to make it easier
        if station_name == "Uhr":
            hours_list = []
            hours = line.replace("  ", " ").replace(" - ", "-")
            hours_space_index = hours.find(" ")
            while hours_space_index != -1:
                if hours_space_index != 0:
                    hours_list.append(hours[:hours_space_index])
                    hours = hours[hours_space_index:]
                else:
                    hours = hours[hours_space_index + 1:]
                hours_space_index = hours.find(" ")
            if hours != "":
                hours_list.append(hours)
            # print(hours_list)
            continue
        else:
            # first entry of line is a space
            makeSchedule(line, hours_list, station_name, schedule)

    # print(schedule)
    # splitting schedule into two lists there and retourn
    for i in range(0, len(schedule) - 1):
        if schedule[i][0] == schedule[i + 1][0]:
            break
        return [schedule[:i + 1], schedule[i + 1:]]


def makeSchedule(line, hours_list, station_name, schedule):
    # Todo anfangs uhrzeit anpassen wenn erste uhrzeit nicht auf alle zutrefend ist siehe 285, dass erste Uhrzeit 502 ist und nicht 402
    current_station = [station_name]

    # appending all departure times
    for hour in hours_list:
        if line == "":
            break
        # there is a range e.g 7-9
        if "-" in hour:
            slash_index = hour.find("-")
            first_hour = int(hour[:slash_index])
            second_hour = int(hour[slash_index + 1:])
            times = []

            # append same minutes in times for all hours in that range
            for i in range(first_hour, second_hour + 1):
                hour_range = i
                if hour_range == first_hour:

                    time_index = 0
                    last_time = -1
                    while time_index < len(line):
                        # minutes always as 2 digit number or -- indicating no minute entry for that hour
                        if line[time_index:time_index + 3] == " --":
                            time_index += 3
                            break

                        time = int(line[time_index:time_index + 3])
                        # checking if minutes out of range
                        if last_time > time:
                            break

                        current_station.append(hour_range * 100 + time)
                        last_time = time
                        times.append(time)
                        # skipping space
                        time_index += 3

                    # delete minutes from line, since they are only for the used hour valid
                    line = line[time_index:]

                else:
                    for time in times:
                        current_station.append(hour_range * 100 + time)

        # there is only one hour
        else:
            hour = int(hour)
            time_index = 0
            last_time = -1
            while time_index < len(line):
                # minutes always as 2 digit number or -- indicating no minute entry for that hour
                if line[time_index:time_index + 3] == " --":
                    time_index += 3
                    break
                # print(int(line[time_index :time_index + 3]))
                time = int(line[time_index:time_index + 3])
                # checking if minutes out of range
                if last_time > time:
                    break

                current_station.append(hour * 100 + time)

                last_time = time
                # skipping space
                time_index += 3

            # delete minutes from line, since they are only for the used hour valid
            line = line[time_index:]

    # default 1 because initialized with station name
    if len(current_station) != 1:
        schedule.append(current_station)
    return


def drawStation(ax, origin, label):
    ax.add_patch(
        Circle(origin, 0.2, fill=True, linewidth=1.2, edgecolor='gray', facecolor="gray", label=label))


def drawEdge(ax, firstPos, secondPos, color):
    ax.add_patch(PathPatch(Path([firstPos, secondPos]), linewidth=1.2, edgecolor=color))


def makeGraph(schedules):
    nodes = []
    graph = np.zeros(shape=(1, 1))
    color = np.chararray(shape=(1, 1), unicode=True)
    j = 0
    colors = "rgby"
    for schedule in schedules[:]:
        for s in schedule:
            last_node = s[0][0]
            if last_node not in nodes:
                nodes += [last_node]

                tmp = np.zeros(shape=(len(nodes), len(nodes)))
                tmp[:tmp.shape[0] - 1, :tmp.shape[0] - 1] = graph
                graph = tmp

                colortmp = np.chararray(shape=(len(nodes), len(nodes)), unicode=True)
                colortmp[:tmp.shape[0] - 1, :tmp.shape[0] - 1] = color
                color = colortmp

            for i in range(1, len(s)):
                new_node = s[i][0]
                if new_node not in nodes:
                    nodes += [new_node]

                    tmp = np.zeros(shape=(len(nodes), len(nodes)))
                    tmp[:(tmp.shape[0] - 1), :tmp.shape[1] - 1] = graph
                    graph = tmp

                    colortmp = np.chararray(shape=(len(nodes), len(nodes)), unicode=True)
                    colortmp[:colortmp.shape[0] - 1, :colortmp.shape[1] - 1] = color
                    color = colortmp

                idx_last = nodes.index(last_node)
                idx_new = nodes.index(new_node)

                # TODO weight passt die berechnung noch nicht! liegt an makeSchedule
                if graph[idx_last][idx_new] == 0:
                    graph[idx_last][idx_new] = s[i][1] - s[i - 1][1]
                    color[idx_last][idx_new] = colors[j]
                last_node = new_node
        j += 1
    return nodes, graph, color


def findMax(graph):
    """
    :param graph: 2-D Array that has all of the connections inside
    :return: index of station, its the index of the row
    """
    max = 0
    idx = 0
    counter = 0

    for i in graph:
        tmp = len(i.nonzero()[0])
        if max < tmp:
            max = tmp
            idx = counter
        counter += 1

    return idx


def draw(ax, schedules):
    # TODO """ """ kommentare
    # Rectangle that contains my routes I want to map
    ax.add_patch(Rectangle((1, 1), 30, 30, fill=False, linewidth=1.2, edgecolor='gray'))

    nodes, graph, color = makeGraph(schedules)
    drawed = np.zeros(shape=(len(nodes), 2))
    firstStation = findMax(graph)
    visited = [False] * len(nodes)

    drawStation(ax, origin=drawed[firstStation], label=nodes[firstStation])

    # hardgecoded
    drawed[firstStation] = 16, 16
    visited[firstStation] = True

    rekDraw(ax, nodes, graph, color, drawed, firstStation, visited)


def rekDraw(ax, nodes, graph, color, drawed, start, visited):
    rotation = 0
    if (len(graph[start].nonzero()[0] != 0)):
        rotation += (2 * np.pi) / len(graph[start].nonzero()[0])
    r = rotation
    visited[start] = True
    for i in range(0, len(graph[start])):

        weight = graph[start][i]

        if weight != 0:
            if all(drawed[i]) == 0:
                origin = drawed[start][0] + np.cos(rotation) * weight, \
                         drawed[start][1] + np.sin(rotation) * weight
                while origin in drawed:
                    rotation += r / 2
                    origin = drawed[start][0] + np.cos(rotation) * weight, \
                             drawed[start][1] + np.sin(rotation) * weight
                drawed[i] = origin

                drawStation(ax, origin, label=nodes[i])
            drawEdge(ax, drawed[start], drawed[i], color[start][i])
            rotation += r
            if not visited[i]:
                rekDraw(ax, nodes, graph, color, drawed, i, visited)


if __name__ == '__main__':
    schedules = []

    line_293 = get_busroute("293")
    line_290 = get_busroute("290")
    line_285 = get_busroute("285")
    line_296 = get_busroute("296")

    schedules.append(line_290)
    # print(line_290)
    schedules.append(line_293)
    schedules.append(line_285)
    # print(line_285)
    schedules.append(line_296)
    # draw(schedules)

    fig, ax = plt.subplots(1)
    fig.tight_layout()
    ax.set_aspect(1)
    ax.axis("off")
    ax.set_xlim(0, 20 * 1.05)
    ax.set_ylim(0, 20 * 1.05)
    draw(ax, schedules)
    # plt.legend()
    plt.show()
