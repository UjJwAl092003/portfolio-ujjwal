export function createMemoryCache() {
  const map = new Map();

  return {
    get(key) {
      return map.get(key);
    },
    set(key, value) {
      map.set(key, value);
    },
    has(key) {
      return map.has(key);
    },
    clear() {
      map.clear();
    },
  };
}
