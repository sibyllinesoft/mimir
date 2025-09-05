/**
 * Unit tests for logging system.
 * 
 * Tests structured logging, different output formats, log levels,
 * and pipeline-specific logging utilities.
 */

import { describe, expect, it, beforeEach, afterEach, spyOn } from 'bun:test';
import { 
  createLogger,
  MimirLogger,
  setupLogging,
  LogLevel 
} from '@/utils/logger';
import type { LoggingConfig } from '@/types';

describe('Logging System', () => {
  let consoleSpy: any;
  let consoleErrorSpy: any;

  beforeEach(() => {
    // Spy on console methods
    consoleSpy = spyOn(console, 'log').mockImplementation(() => {});
    consoleErrorSpy = spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    // Restore console methods
    consoleSpy.mockRestore();
    consoleErrorSpy.mockRestore();
  });

  describe('Logger Creation', () => {
    it('should create logger with default configuration', () => {
      const logger = createLogger('test-logger');
      
      expect(logger).toBeInstanceOf(MimirLogger);
      expect((logger as MimirLogger).name).toBe('test-logger');
    });

    it('should create logger with custom configuration', () => {
      const config: LoggingConfig = {
        logLevel: 'debug',
        logFormat: 'human',
        logIncludeTimestamp: true,
        logIncludeLevel: true,
        logIncludeLogger: true,
        logIncludeThread: false,
        logMaxSize: '10m',
        logMaxFiles: 5
      };

      const logger = createLogger('custom-logger', config);
      
      expect(logger).toBeInstanceOf(MimirLogger);
      expect((logger as MimirLogger).name).toBe('custom-logger');
    });
  });

  describe('Log Levels', () => {
    it('should log debug messages when level is debug', () => {
      const config: LoggingConfig = {
        logLevel: 'debug',
        logFormat: 'human',
        logIncludeTimestamp: false,
        logIncludeLevel: true,
        logIncludeLogger: true,
        logIncludeThread: false,
        logMaxSize: '10m',
        logMaxFiles: 5
      };

      const logger = createLogger('debug-test', config);
      
      logger.debug('Debug message');
      logger.info('Info message');
      logger.warn('Warning message');
      logger.error('Error message');

      expect(consoleSpy).toHaveBeenCalledTimes(4);
    });

    it('should respect info log level', () => {
      const config: LoggingConfig = {
        logLevel: 'info',
        logFormat: 'human',
        logIncludeTimestamp: false,
        logIncludeLevel: true,
        logIncludeLogger: true,
        logIncludeThread: false,
        logMaxSize: '10m',
        logMaxFiles: 5
      };

      const logger = createLogger('info-test', config);
      
      logger.debug('Debug message'); // Should not log
      logger.info('Info message');
      logger.warn('Warning message');
      logger.error('Error message');

      expect(consoleSpy).toHaveBeenCalledTimes(3); // info, warn, error
    });

    it('should respect warn log level', () => {
      const config: LoggingConfig = {
        logLevel: 'warn',
        logFormat: 'human',
        logIncludeTimestamp: false,
        logIncludeLevel: true,
        logIncludeLogger: true,
        logIncludeThread: false,
        logMaxSize: '10m',
        logMaxFiles: 5
      };

      const logger = createLogger('warn-test', config);
      
      logger.debug('Debug message'); // Should not log
      logger.info('Info message');   // Should not log
      logger.warn('Warning message');
      logger.error('Error message');

      expect(consoleSpy).toHaveBeenCalledTimes(1); // warn only
      expect(consoleErrorSpy).toHaveBeenCalledTimes(1); // error only
    });

    it('should respect error log level', () => {
      const config: LoggingConfig = {
        logLevel: 'error',
        logFormat: 'human',
        logIncludeTimestamp: false,
        logIncludeLevel: true,
        logIncludeLogger: true,
        logIncludeThread: false,
        logMaxSize: '10m',
        logMaxFiles: 5
      };

      const logger = createLogger('error-test', config);
      
      logger.debug('Debug message'); // Should not log
      logger.info('Info message');   // Should not log
      logger.warn('Warning message'); // Should not log
      logger.error('Error message');

      expect(consoleSpy).toHaveBeenCalledTimes(0);
      expect(consoleErrorSpy).toHaveBeenCalledTimes(1); // error only
    });
  });

  describe('Log Formats', () => {
    it('should format logs as JSON', () => {
      const config: LoggingConfig = {
        logLevel: 'info',
        logFormat: 'json',
        logIncludeTimestamp: true,
        logIncludeLevel: true,
        logIncludeLogger: true,
        logIncludeThread: false,
        logMaxSize: '10m',
        logMaxFiles: 5
      };

      const logger = createLogger('json-test', config);
      logger.info('Test message', { key: 'value' });

      expect(consoleSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleSpy.mock.calls[0][0];
      
      // Should be valid JSON
      expect(() => JSON.parse(loggedMessage)).not.toThrow();
      
      const parsed = JSON.parse(loggedMessage);
      expect(parsed.level).toBe('info');
      expect(parsed.message).toBe('Test message');
      expect(parsed.logger).toBe('json-test');
      expect(parsed.meta.key).toBe('value');
      expect(parsed.timestamp).toBeDefined();
    });

    it('should format logs as human readable', () => {
      const config: LoggingConfig = {
        logLevel: 'info',
        logFormat: 'human',
        logIncludeTimestamp: false,
        logIncludeLevel: true,
        logIncludeLogger: true,
        logIncludeThread: false,
        logMaxSize: '10m',
        logMaxFiles: 5
      };

      const logger = createLogger('human-test', config);
      logger.info('Test message');

      expect(consoleSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleSpy.mock.calls[0][0];
      
      expect(loggedMessage).toContain('INFO');
      expect(loggedMessage).toContain('human-test');
      expect(loggedMessage).toContain('Test message');
    });

    it('should format logs as plain text', () => {
      const config: LoggingConfig = {
        logLevel: 'info',
        logFormat: 'text',
        logIncludeTimestamp: false,
        logIncludeLevel: false,
        logIncludeLogger: false,
        logIncludeThread: false,
        logMaxSize: '10m',
        logMaxFiles: 5
      };

      const logger = createLogger('text-test', config);
      logger.info('Test message');

      expect(consoleSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleSpy.mock.calls[0][0];
      
      expect(loggedMessage).toBe('Test message');
    });
  });

  describe('Log Components', () => {
    it('should include timestamp when configured', () => {
      const config: LoggingConfig = {
        logLevel: 'info',
        logFormat: 'human',
        logIncludeTimestamp: true,
        logIncludeLevel: true,
        logIncludeLogger: true,
        logIncludeThread: false,
        logMaxSize: '10m',
        logMaxFiles: 5
      };

      const logger = createLogger('timestamp-test', config);
      logger.info('Test message');

      expect(consoleSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleSpy.mock.calls[0][0];
      
      // Should contain an ISO timestamp pattern
      expect(loggedMessage).toMatch(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
    });

    it('should exclude timestamp when configured', () => {
      const config: LoggingConfig = {
        logLevel: 'info',
        logFormat: 'human',
        logIncludeTimestamp: false,
        logIncludeLevel: true,
        logIncludeLogger: true,
        logIncludeThread: false,
        logMaxSize: '10m',
        logMaxFiles: 5
      };

      const logger = createLogger('no-timestamp-test', config);
      logger.info('Test message');

      expect(consoleSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleSpy.mock.calls[0][0];
      
      // Should not contain timestamp pattern
      expect(loggedMessage).not.toMatch(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
    });

    it('should include logger name when configured', () => {
      const config: LoggingConfig = {
        logLevel: 'info',
        logFormat: 'human',
        logIncludeTimestamp: false,
        logIncludeLevel: true,
        logIncludeLogger: true,
        logIncludeThread: false,
        logMaxSize: '10m',
        logMaxFiles: 5
      };

      const logger = createLogger('logger-name-test', config);
      logger.info('Test message');

      expect(consoleSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleSpy.mock.calls[0][0];
      
      expect(loggedMessage).toContain('logger-name-test');
    });

    it('should exclude logger name when configured', () => {
      const config: LoggingConfig = {
        logLevel: 'info',
        logFormat: 'human',
        logIncludeTimestamp: false,
        logIncludeLevel: true,
        logIncludeLogger: false,
        logIncludeThread: false,
        logMaxSize: '10m',
        logMaxFiles: 5
      };

      const logger = createLogger('no-logger-name-test', config);
      logger.info('Test message');

      expect(consoleSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleSpy.mock.calls[0][0];
      
      expect(loggedMessage).not.toContain('no-logger-name-test');
    });
  });

  describe('Metadata Logging', () => {
    it('should include metadata in JSON format', () => {
      const config: LoggingConfig = {
        logLevel: 'info',
        logFormat: 'json',
        logIncludeTimestamp: false,
        logIncludeLevel: true,
        logIncludeLogger: true,
        logIncludeThread: false,
        logMaxSize: '10m',
        logMaxFiles: 5
      };

      const logger = createLogger('meta-test', config);
      const metadata = {
        userId: '12345',
        action: 'file_upload',
        fileSize: 1024,
        nested: {
          property: 'value'
        }
      };

      logger.info('User action performed', metadata);

      expect(consoleSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleSpy.mock.calls[0][0];
      const parsed = JSON.parse(loggedMessage);
      
      expect(parsed.meta.userId).toBe('12345');
      expect(parsed.meta.action).toBe('file_upload');
      expect(parsed.meta.fileSize).toBe(1024);
      expect(parsed.meta.nested.property).toBe('value');
    });

    it('should handle metadata in human format', () => {
      const config: LoggingConfig = {
        logLevel: 'info',
        logFormat: 'human',
        logIncludeTimestamp: false,
        logIncludeLevel: true,
        logIncludeLogger: true,
        logIncludeThread: false,
        logMaxSize: '10m',
        logMaxFiles: 5
      };

      const logger = createLogger('meta-human-test', config);
      const metadata = { key: 'value', count: 42 };

      logger.info('Test with metadata', metadata);

      expect(consoleSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleSpy.mock.calls[0][0];
      
      expect(loggedMessage).toContain('Test with metadata');
      expect(loggedMessage).toContain('key: value');
      expect(loggedMessage).toContain('count: 42');
    });

    it('should handle complex metadata objects', () => {
      const config: LoggingConfig = {
        logLevel: 'info',
        logFormat: 'json',
        logIncludeTimestamp: false,
        logIncludeLevel: true,
        logIncludeLogger: true,
        logIncludeThread: false,
        logMaxSize: '10m',
        logMaxFiles: 5
      };

      const logger = createLogger('complex-meta-test', config);
      const complexMeta = {
        array: [1, 2, 3],
        object: { nested: true },
        date: new Date('2024-01-01'),
        null: null,
        undefined: undefined
      };

      logger.info('Complex metadata test', complexMeta);

      expect(consoleSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleSpy.mock.calls[0][0];
      const parsed = JSON.parse(loggedMessage);
      
      expect(parsed.meta.array).toEqual([1, 2, 3]);
      expect(parsed.meta.object.nested).toBe(true);
      expect(parsed.meta.null).toBeNull();
      expect(parsed.meta.undefined).toBeUndefined();
    });
  });

  describe('Error Logging', () => {
    it('should log errors with stack traces', () => {
      const config: LoggingConfig = {
        logLevel: 'error',
        logFormat: 'json',
        logIncludeTimestamp: false,
        logIncludeLevel: true,
        logIncludeLogger: true,
        logIncludeThread: false,
        logMaxSize: '10m',
        logMaxFiles: 5
      };

      const logger = createLogger('error-test', config);
      const error = new Error('Test error');
      
      logger.error('An error occurred', { error });

      expect(consoleErrorSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleErrorSpy.mock.calls[0][0];
      const parsed = JSON.parse(loggedMessage);
      
      expect(parsed.level).toBe('error');
      expect(parsed.message).toBe('An error occurred');
      expect(parsed.meta.error.message).toBe('Test error');
      expect(parsed.meta.error.stack).toBeDefined();
    });

    it('should handle Error objects directly', () => {
      const config: LoggingConfig = {
        logLevel: 'error',
        logFormat: 'human',
        logIncludeTimestamp: false,
        logIncludeLevel: true,
        logIncludeLogger: true,
        logIncludeThread: false,
        logMaxSize: '10m',
        logMaxFiles: 5
      };

      const logger = createLogger('direct-error-test', config);
      const error = new Error('Direct error');
      
      logger.error('Error occurred', error);

      expect(consoleErrorSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleErrorSpy.mock.calls[0][0];
      
      expect(loggedMessage).toContain('Error occurred');
      expect(loggedMessage).toContain('Direct error');
    });
  });

  describe('Setup Logging', () => {
    it('should configure global logging', () => {
      const config: LoggingConfig = {
        logLevel: 'warn',
        logFormat: 'json',
        logIncludeTimestamp: true,
        logIncludeLevel: true,
        logIncludeLogger: true,
        logIncludeThread: false,
        logMaxSize: '50m',
        logMaxFiles: 10
      };

      // Should not throw
      expect(() => setupLogging(config)).not.toThrow();
    });

    it('should handle missing configuration gracefully', () => {
      // Should not throw with undefined config
      expect(() => setupLogging(undefined as any)).not.toThrow();
    });
  });

  describe('Log Level Numeric Values', () => {
    it('should have correct log level hierarchy', () => {
      expect(LogLevel.DEBUG).toBeLessThan(LogLevel.INFO);
      expect(LogLevel.INFO).toBeLessThan(LogLevel.WARN);
      expect(LogLevel.WARN).toBeLessThan(LogLevel.ERROR);
    });
  });
});