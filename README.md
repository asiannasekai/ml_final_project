# Mycelial Router

A bio-inspired routing algorithm based on fungal growth patterns, designed to optimize for both resource collection and resilience in network routing.

## Features

- Simulates mycelial (fungal) growth patterns for network routing
- Implements greedy growth with local nutrient sensing
- Supports obstacles and node failures
- Compares performance with Dijkstra's algorithm
- Visualizes growth process and network state
- Measures path redundancy and failure tolerance

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/mycelial-router.git
cd mycelial-router
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main implementation is in `mycelial_router/core.py`. To run the test examples:

```bash
python -m mycelial_router.test
```

This will demonstrate:
1. Basic mycelial growth on a small grid
2. Growth with node failures
3. Comparison with Dijkstra's algorithm

## Algorithm Overview

1. **Grid Initialization**:
   - Creates a 2D grid with random nutrient values and traversal costs
   - Adds obstacles and potential node failures
   - Converts grid to a network graph

2. **Mycelial Growth**:
   - Starts from a seed node
   - Grows towards neighbors based on:
     - Nutrient value (reward)
     - Traversal cost (inverse weight)
   - Supports branching for exploration
   - Avoids obstacles and failed nodes

3. **Path Comparison**:
   - Compares mycelial paths with Dijkstra's algorithm
   - Measures:
     - Total path cost
     - Path redundancy
     - Failure tolerance

## Parameters

- `grid_size`: Size of the routing grid
- `nutrient_range`: Range of nutrient values
- `cost_range`: Range of traversal costs
- `obstacle_prob`: Probability of obstacles
- `failure_prob`: Probability of node failures

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 