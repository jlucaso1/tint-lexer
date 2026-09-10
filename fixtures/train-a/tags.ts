type Tag = { name: string; weight: number };

const MIN_WEIGHT = 3;

function selectTags(tags: Tag[]): string[] {
  // Keep the input order for equally weighted tags.
  const selected: string[] = [];
  for (const tag of tags) {
    if (tag.weight >= MIN_WEIGHT) {
      selected.push(`#${tag.name}`);
    }
  }
  return selected;
}

const tags = [
  { name: "garden", weight: 5 },
  { name: "desk", weight: 1 },
];
console.log(selectTags(tags).join(", "));
