/**
 * Test setup and global configuration.
 * 
 * Configures test environment, mocks, and global utilities
 * for the Mimir TypeScript test suite.
 */

// Set test environment
process.env.NODE_ENV = 'test';

// Mock environment variables for consistent testing
process.env.MIMIR_LOG_LEVEL = 'error'; // Reduce noise during tests
process.env.LENS_ENABLED = 'false'; // Disable Lens by default in tests
process.env.MIMIR_ENABLE_METRICS = 'false';
process.env.MIMIR_ENABLE_PROFILING = 'false';

// Configure test timeouts
// Note: Bun specific configurations would go here if needed

// Global test utilities
declare global {
  namespace jest {
    interface Matchers<R> {
      toBeWithinRange(a: number, b: number): R;
    }
  }
}

// Custom matchers
expect.extend({
  toBeWithinRange(received: number, floor: number, ceiling: number) {
    const pass = received >= floor && received <= ceiling;
    if (pass) {
      return {
        message: () =>
          `expected ${received} not to be within range ${floor} - ${ceiling}`,
        pass: true,
      };
    } else {
      return {
        message: () =>
          `expected ${received} to be within range ${floor} - ${ceiling}`,
        pass: false,
      };
    }
  },
});

// Console cleanup for tests
const originalConsole = console;
global.console = {
  ...originalConsole,
  // Suppress logs during tests unless explicitly enabled
  log: process.env.TEST_VERBOSE === 'true' ? originalConsole.log : () => {},
  info: process.env.TEST_VERBOSE === 'true' ? originalConsole.info : () => {},
  warn: process.env.TEST_VERBOSE === 'true' ? originalConsole.warn : () => {},
  error: originalConsole.error, // Keep errors visible
};

// Unhandled promise rejection handling
process.on('unhandledRejection', (reason) => {
  console.error('Unhandled promise rejection in tests:', reason);
  process.exit(1);
});

// Test timeout warning
const originalSetTimeout = setTimeout;
global.setTimeout = ((fn: Function, delay: number, ...args: any[]) => {
  if (delay > 10000) {
    console.warn(`Long timeout detected in tests: ${delay}ms`);
  }
  return originalSetTimeout(fn, delay, ...args);
}) as typeof setTimeout;

// Mock timers utility
export const mockTimers = {
  useFakeTimers: () => {
    // Implement fake timers if needed
  },
  useRealTimers: () => {
    // Restore real timers
  },
  advanceTimersByTime: (ms: number) => {
    // Advance fake timers
  }
};

// Test data utilities
export const createTestDirectory = (name: string = 'test') => {
  const { tmpdir } = require('os');
  const { join } = require('path');
  const { mkdirSync } = require('fs');
  
  const dir = join(tmpdir(), `mimir-test-${name}-${Date.now()}`);
  mkdirSync(dir, { recursive: true });
  return dir;
};

export const createTestFile = (dir: string, name: string, content: string) => {
  const { join } = require('path');
  const { writeFileSync } = require('fs');
  
  const filePath = join(dir, name);
  writeFileSync(filePath, content);
  return filePath;
};

// Cleanup utilities
export const cleanup = (paths: string[]) => {
  const { rmSync } = require('fs');
  
  paths.forEach(path => {
    try {
      rmSync(path, { recursive: true, force: true });
    } catch (error) {
      // Ignore cleanup errors
    }
  });
};

// Mock HTTP requests for testing
export const mockFetch = (responses: Array<{ url?: string; response: any }>) => {
  const originalFetch = global.fetch;
  let callCount = 0;
  
  global.fetch = jest.fn().mockImplementation((url: string) => {
    const response = responses[callCount] || responses[responses.length - 1];
    callCount++;
    
    if (response.url && !url.includes(response.url)) {
      return Promise.reject(new Error(`Unexpected URL: ${url}`));
    }
    
    return Promise.resolve({
      ok: response.response.ok !== false,
      status: response.response.status || 200,
      json: () => Promise.resolve(response.response.json || {}),
      text: () => Promise.resolve(response.response.text || ''),
    });
  });
  
  return () => {
    global.fetch = originalFetch;
  };
};

console.log('🧪 Test environment initialized');