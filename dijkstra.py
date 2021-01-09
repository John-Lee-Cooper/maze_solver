""" Provide an Abstract network and node class that implements the Dijksta algorithm """

from abc import ABC, abstractmethod
from heapq import heappop, heappush
from typing import Dict, Generator, Tuple


class Node:
    """ Node class for dijkstra """

    def __init__(self):

        self.explored = False
        self.prev = None
        self.cost = float("inf")  # distance from source

    def __lt__(self, other):
        return self.cost < other.cost

    def path(self) -> Generator["Node", "Node", None]:
        """ Generate a path containing this node and all its ancestors """

        node = self
        while node.prev:
            yield node
            node = node.prev
        yield node


class Network(ABC):
    """ Abstract network class for dijkstra """

    def __init__(self, nodes: Dict[Tuple, Node]):
        self.nodes = nodes

    def find_shortest_path(
        self, src_node: Node, dst_node: Node
    ) -> Generator[Node, Node, None]:
        """
        Use dijkstra to find the shortest path from src_node to dst_node and
        return the path

        Yields Generator[yield_type: Node, send_type: Node, return_type: None]
        """
        if self._dijkstra(src_node, dst_node):
            return dst_node.path()
        return None

    def _dijkstra(self, src_node: Node, dst_node: Node) -> bool:
        """
        Dijkstra algorithm: use a priority queue (pq) to explore Network
        starting at src_node and terminating when dst_node is found
        or network is exhausted
        """
        src_node.cost = 0
        pq = [src_node]

        while pq:
            curr = heappop(pq)
            curr.explored = True

            for nxt_, cost in self._neighbors(curr):
                if nxt_.explored:
                    continue
                cost += curr.cost
                if cost >= nxt_.cost:
                    continue
                nxt_.cost = cost
                nxt_.prev = curr
                heappush(pq, nxt_)
                self._display(nxt_)
                if nxt_ is dst_node:
                    return True
        return False

    @abstractmethod
    def _neighbors(self, node: Node) -> Tuple[Node, int]:
        """ Yield neighbors to node, and the cost to get there """

    @abstractmethod
    def _display(self, curr_node: Node) -> None:
        """ Periodically draw the current best solution """
