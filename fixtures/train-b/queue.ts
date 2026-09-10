interface Job {
  id: number;
  ready: boolean;
}

function nextJob(jobs: Job[]): Job | undefined {
  // Leave blocked jobs in the queue.
  const index = jobs.findIndex(job => job.ready);
  if (index < 0) return undefined;
  return jobs.splice(index, 1)[0];
}

const pending: Job[] = [
  { id: 11, ready: false },
  { id: 12, ready: true },
];
const job = nextJob(pending);
console.log(job ? `running ${job.id}` : "idle");
