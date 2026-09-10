export interface Ticket {
  name: string;
  priority: number;
}

export function urgent(tickets: Ticket[]): string[] {
  // Priority zero means the ticket can wait.
  const ordered = [...tickets].sort((a, b) => b.priority - a.priority);
  return ordered.filter(t => t.priority > 0).map(t => t.name);
}

const backlog: Ticket[] = [
  { name: "repair", priority: 2 },
  { name: "paint", priority: 0 },
];

for (const name of urgent(backlog)) {
  console.log(`dispatch ${name}`);
}
