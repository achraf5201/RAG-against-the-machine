document1 = """The machine is designed for industrial manufacturing and automated production. This machine can process materials quickly while maintaining consistent quality. Engineers inspect the machine regularly to ensure safe operation, reduce maintenance costs, and improve overall performance. Modern factories use advanced equipment to increase productivity and minimize production errors."""

document2 = """The machine operates in a modern factory where workers assemble electronic components. Automated sensors monitor temperature, speed, and pressure throughout the production process. Operators receive alerts when performance changes unexpectedly. Regular maintenance helps prevent breakdowns and keeps the equipment running efficiently during long production shifts."""

document3 = """Every manufacturing company needs reliable equipment to maintain product quality. The machine helps employees complete repetitive tasks with greater precision. Production managers analyze performance data to identify bottlenecks and improve workflow. Proper training also helps workers operate equipment safely and avoid unnecessary delays during daily manufacturing activities."""

document4 = """Industrial automation has transformed how products are manufactured around the world. Factories now combine robotics, sensors, software, and mechanical equipment to produce goods more efficiently. A machine can perform repetitive operations with consistent accuracy, helping businesses reduce defects and improve productivity. Engineers design production systems that coordinate multiple operations while maintaining strict quality standards.

In a modern facility, each machine is connected to monitoring software that records operating conditions. These measurements help technicians detect unusual vibration, excessive heat, or changes in production speed. Predictive maintenance systems analyze this information to estimate when components may require inspection or replacement. This approach can reduce unexpected downtime and extend equipment life.

Manufacturing teams also focus on workplace safety. Employees receive training on operating procedures, emergency stops, and protective equipment. Supervisors review production reports and investigate problems that could affect product quality. When equipment is maintained correctly, factories can meet delivery schedules and control operating expenses.

New technologies continue to improve industrial processes. Digital simulations allow engineers to test production layouts before installing physical equipment. Data analysis helps managers compare performance across production lines and identify opportunities for improvement. By combining skilled employees with reliable technology, manufacturers can respond to changing demand while maintaining consistent standards and efficient operations."""

print(
    document1.count("machine"),
    document2.count("machine"),
    document3.count("machine"),
    document4.count("machine"),
)


import bm25s

documents = [
    document1,
    document2,
    document3,
    document4,
]

corpus_tokens = bm25s.tokenize(documents)
print(corpus_tokens)

retriever = bm25s.BM25()
retriever.index(corpus_tokens)

query = "How can I maintain a machine and prevent breakdowns?"
query_tokens = bm25s.tokenize(query)

results, scores = retriever.retrieve(
    query_tokens,
    k=2,
    corpus=documents
)


for i in range(results.shape[1]):
    print(f"Rank: {i + 1}")
    print(f"Score: {scores[0, i]}")
    print(f"Document: {results[0, i]}")
    print("-" * 50)
