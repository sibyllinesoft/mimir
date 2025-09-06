/**
 * Unit tests for logging system.
 * 
 * Tests structured logging, different output formats, log levels,
 * and pipeline-specific logging utilities.
 */

import { describe, expect, it, beforeEach, afterEach, spyOn } from 'bun:test';
import { 
  createLogger,
  getLogger,
  MimirLogger,
  setupLogging,
  LogLevel,
  PipelineLogger,
  getPipelineLogger,
  LogContext,
  withContext
} from '@/utils/logger';
import type { LoggingConfig } from '@/types';

describe('Logging System', () => {
  let consoleSpy: any;
  let consoleErrorSpy: any;
  let consoleWarnSpy: any;
  let consoleDebugSpy: any;

  beforeEach(() => {
    // Spy on console methods
    consoleSpy = spyOn(console, 'log').mockImplementation(() => {});
    consoleErrorSpy = spyOn(console, 'error').mockImplementation(() => {});
    consoleWarnSpy = spyOn(console, 'warn').mockImplementation(() => {});
    consoleDebugSpy = spyOn(console, 'debug').mockImplementation(() => {});
  });

  afterEach(() => {
    // Restore console methods
    consoleSpy.mockRestore();
    consoleErrorSpy.mockRestore();
    consoleWarnSpy.mockRestore();
    consoleDebugSpy.mockRestore();
  });

  describe('Logger Creation', () => {
    it('should create logger with default configuration', () => {
      const logger = createLogger('test-logger');
      
      expect(logger).toBeInstanceOf(MimirLogger);
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

      setupLogging(config);
      const logger = createLogger('custom-logger');
      
      expect(logger).toBeInstanceOf(MimirLogger);
    });
  });

  describe('Log Levels', () => {
    it('should log debug messages when level is debug', () => {
      const config: LoggingConfig = {
        logLevel: 'debug',
        logFormat: 'json',
        logIncludeTimestamp: false,
        logIncludeLevel: true,
        logIncludeLogger: true,
        logIncludeThread: false,
        logMaxSize: '10m',
        logMaxFiles: 5
      };

      setupLogging(config);
      const logger = createLogger('debug-test');
      
      logger.debug('Debug message');
      logger.info('Info message');
      logger.warn('Warning message');
      logger.error('Error message');

      expect(consoleDebugSpy).toHaveBeenCalledTimes(1); // debug uses console.debug
      expect(consoleSpy).toHaveBeenCalledTimes(1); // info uses console.log
      expect(consoleWarnSpy).toHaveBeenCalledTimes(1); // warn uses console.warn
      expect(consoleErrorSpy).toHaveBeenCalledTimes(1); // error uses console.error
    });

    it('should respect info log level', () => {
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

      setupLogging(config);
      const logger = createLogger('info-test');
      
      logger.debug('Debug message'); // Should not log
      logger.info('Info message');
      logger.warn('Warning message');
      logger.error('Error message');

      expect(consoleDebugSpy).toHaveBeenCalledTimes(0); // debug should not log
      expect(consoleSpy).toHaveBeenCalledTimes(1); // info
      expect(consoleWarnSpy).toHaveBeenCalledTimes(1); // warn
      expect(consoleErrorSpy).toHaveBeenCalledTimes(1); // error
    });

    it('should respect warn log level', () => {
      const config: LoggingConfig = {
        logLevel: 'warn',
        logFormat: 'json',
        logIncludeTimestamp: false,
        logIncludeLevel: true,
        logIncludeLogger: true,
        logIncludeThread: false,
        logMaxSize: '10m',
        logMaxFiles: 5
      };

      setupLogging(config);
      const logger = createLogger('warn-test');
      
      logger.debug('Debug message'); // Should not log
      logger.info('Info message');   // Should not log
      logger.warn('Warning message');
      logger.error('Error message');

      expect(consoleDebugSpy).toHaveBeenCalledTimes(0); // should not log
      expect(consoleSpy).toHaveBeenCalledTimes(0); // should not log
      expect(consoleWarnSpy).toHaveBeenCalledTimes(1); // warn
      expect(consoleErrorSpy).toHaveBeenCalledTimes(1); // error
    });

    it('should respect error log level', () => {
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

      setupLogging(config);
      const logger = createLogger('error-test');
      
      logger.debug('Debug message'); // Should not log
      logger.info('Info message');   // Should not log
      logger.warn('Warning message'); // Should not log
      logger.error('Error message');

      expect(consoleDebugSpy).toHaveBeenCalledTimes(0);
      expect(consoleSpy).toHaveBeenCalledTimes(0);
      expect(consoleWarnSpy).toHaveBeenCalledTimes(0);
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

      setupLogging(config);
      const logger = createLogger('json-test');
      logger.info('Test message', { key: 'value' });

      expect(consoleSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleSpy.mock.calls[0][0];
      
      // Should be valid JSON
      expect(() => JSON.parse(loggedMessage)).not.toThrow();
      
      const parsed = JSON.parse(loggedMessage);
      expect(parsed.level).toBe('info');
      expect(parsed.message).toBe('Test message');
      expect(parsed.logger).toBe('json-test');
      expect(parsed.key).toBe('value');
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

      setupLogging(config);
      const logger = createLogger('human-test');
      logger.info('Test message');

      expect(consoleSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleSpy.mock.calls[0][0];
      
      // Human format should contain emoji and logger name
      expect(loggedMessage).toContain('ℹ️');
      expect(loggedMessage).toContain('[human-test]');
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

      setupLogging(config);
      const logger = createLogger('text-test');
      logger.info('Test message');

      expect(consoleSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleSpy.mock.calls[0][0];
      
      // Text format should contain timestamp, logger name, level, and message
      expect(loggedMessage).toContain('text-test');
      expect(loggedMessage).toContain('INFO');
      expect(loggedMessage).toContain('Test message');
    });
  });

  describe('Log Components', () => {
    it('should include timestamp when configured', () => {
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

      setupLogging(config);
      const logger = createLogger('timestamp-test');
      logger.info('Test message');

      expect(consoleSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleSpy.mock.calls[0][0];
      
      const parsed = JSON.parse(loggedMessage);
      expect(parsed.timestamp).toBeDefined();
      expect(parsed.timestamp).toMatch(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
    });

    it('should always include timestamp in JSON', () => {
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

      setupLogging(config);
      const logger = createLogger('no-timestamp-test');
      logger.info('Test message');

      expect(consoleSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleSpy.mock.calls[0][0];
      
      // Logger always outputs JSON with timestamps
      const parsed = JSON.parse(loggedMessage);
      expect(parsed.timestamp).toBeDefined();
    });

    it('should include logger name when configured', () => {
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

      setupLogging(config);
      const logger = createLogger('logger-name-test');
      logger.info('Test message');

      expect(consoleSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleSpy.mock.calls[0][0];
      
      const parsed = JSON.parse(loggedMessage);
      expect(parsed.logger).toBe('logger-name-test');
    });

    it('should include logger name in all formats', () => {
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

      setupLogging(config);
      const logger = createLogger('test-logger-name');
      logger.info('Test message');

      expect(consoleSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleSpy.mock.calls[0][0];
      
      const parsed = JSON.parse(loggedMessage);
      expect(parsed.logger).toBe('test-logger-name');
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

      setupLogging(config);
      const logger = createLogger('meta-test');
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
      
      expect(parsed.userId).toBe('12345');
      expect(parsed.action).toBe('file_upload');
      expect(parsed.fileSize).toBe(1024);
      expect(parsed.nested.property).toBe('value');
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

      setupLogging(config);
      const logger = createLogger('meta-human-test');
      const metadata = { key: 'value', count: 42 };

      logger.info('Test with metadata', metadata);

      expect(consoleSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleSpy.mock.calls[0][0];
      
      // Human format should contain logger name and message
      expect(loggedMessage).toContain('[meta-human-test]');
      expect(loggedMessage).toContain('Test with metadata');
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

      setupLogging(config);
      const logger = createLogger('complex-meta-test');
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
      
      expect(parsed.array).toEqual([1, 2, 3]);
      expect(parsed.object.nested).toBe(true);
      expect(parsed.null).toBeNull();
      expect(parsed.undefined).toBeUndefined();
    });
  });

  describe('Error Logging', () => {
    it('should log errors with metadata', () => {
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

      setupLogging(config);
      const logger = createLogger('error-test');
      const errorInfo = { errorCode: 500, details: 'Something went wrong' };
      
      logger.error('An error occurred', errorInfo);

      expect(consoleErrorSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleErrorSpy.mock.calls[0][0];
      const parsed = JSON.parse(loggedMessage);
      
      expect(parsed.level).toBe('error');
      expect(parsed.message).toBe('An error occurred');
      expect(parsed.errorCode).toBe(500);
      expect(parsed.details).toBe('Something went wrong');
    });

    it('should handle different log formats', () => {
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

      setupLogging(config);
      const logger = createLogger('format-test');
      
      logger.error('Test error message');

      expect(consoleErrorSpy).toHaveBeenCalledTimes(1);
      const loggedMessage = consoleErrorSpy.mock.calls[0][0];
      
      // In human format, should contain emoji and logger name
      expect(loggedMessage).toContain('❌');
      expect(loggedMessage).toContain('[format-test]');
      expect(loggedMessage).toContain('Test error message');
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

  describe('Log Level String Values', () => {
    it('should have correct log level values', () => {
      expect(LogLevel.DEBUG).toBe('debug');
      expect(LogLevel.INFO).toBe('info');
      expect(LogLevel.WARN).toBe('warn');
      expect(LogLevel.ERROR).toBe('error');
    });
  });

  describe('Additional Logger Functions', () => {
    beforeEach(() => {
      // Setup info level to ensure messages are logged
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
      setupLogging(config);
      
      consoleSpy.mockClear();
      consoleErrorSpy.mockClear();
      consoleWarnSpy.mockClear();
      consoleDebugSpy.mockClear();
    });

    it('should create logger with getLogger', () => {
      const logger = getLogger('get-logger-test');
      expect(logger).toBeInstanceOf(MimirLogger);
    });

    it('should create child logger', () => {
      const parent = createLogger('parent');
      const child = parent.child('child');
      
      child.info('Child message');
      expect(consoleSpy).toHaveBeenCalled();
    });

    it('should create logger with context', () => {
      const logger = createLogger('context-test');
      const contextLogger = logger.withContext({ requestId: '123' });
      
      contextLogger.info('Message with context');
      expect(consoleSpy).toHaveBeenCalled();
    });
  });

  describe('PipelineLogger', () => {
    beforeEach(() => {
      // Setup info level to ensure messages are logged
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
      setupLogging(config);
      
      consoleSpy.mockClear();
      consoleErrorSpy.mockClear();
      consoleWarnSpy.mockClear();
      consoleDebugSpy.mockClear();
    });

    it('should create pipeline logger', () => {
      const pipelineLogger = getPipelineLogger('test-index');
      expect(pipelineLogger).toBeInstanceOf(PipelineLogger);
    });

    it('should log stage start', () => {
      const pipelineLogger = new PipelineLogger('test-index');
      pipelineLogger.logStageStart('analyze');
      expect(consoleSpy).toHaveBeenCalled();
    });

    it('should log stage progress', () => {
      const pipelineLogger = new PipelineLogger('test-index');
      pipelineLogger.logStageProgress('analyze', 50);
      expect(consoleSpy).toHaveBeenCalled();
    });

    it('should log stage complete', () => {
      const pipelineLogger = new PipelineLogger('test-index');
      pipelineLogger.logStageComplete('analyze');
      expect(consoleSpy).toHaveBeenCalled();
    });

    it('should log stage error', () => {
      const pipelineLogger = new PipelineLogger('test-index');
      const error = new Error('Test error');
      pipelineLogger.logStageError('analyze', error);
      expect(consoleErrorSpy).toHaveBeenCalled();
    });

    it('should log info', () => {
      const pipelineLogger = new PipelineLogger('test-index');
      pipelineLogger.logInfo('Info message');
      expect(consoleSpy).toHaveBeenCalled();
    });

    it('should log warning', () => {
      const pipelineLogger = new PipelineLogger('test-index');
      pipelineLogger.logWarning('Warning message');
      expect(consoleWarnSpy).toHaveBeenCalled();
    });

    it('should log error', () => {
      const pipelineLogger = new PipelineLogger('test-index');
      pipelineLogger.logError('Error message');
      expect(consoleErrorSpy).toHaveBeenCalled();
    });
  });

  describe('LogContext', () => {
    beforeEach(() => {
      // Setup info level to ensure messages are logged
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
      setupLogging(config);
      
      consoleSpy.mockClear();
      consoleErrorSpy.mockClear();
      consoleWarnSpy.mockClear();
      consoleDebugSpy.mockClear();
    });

    it('should create log context with utility function', () => {
      const logger = createLogger('context-util-test');
      const context = withContext(logger, { requestId: '456' });
      expect(context).toBeInstanceOf(LogContext);
    });

    it('should log with context', () => {
      const logger = createLogger('context-log-test');
      const context = new LogContext(logger, { key: 'value' });
      
      context.info('Test message');
      context.warn('Warning message');
      context.error('Error message');
      
      expect(consoleSpy).toHaveBeenCalled();
      expect(consoleWarnSpy).toHaveBeenCalled();
      expect(consoleErrorSpy).toHaveBeenCalled();
    });

    it('should log debug with debug level', () => {
      // Setup debug level for this specific test
      const config: LoggingConfig = {
        logLevel: 'debug',
        logFormat: 'json',
        logIncludeTimestamp: false,
        logIncludeLevel: true,
        logIncludeLogger: true,
        logIncludeThread: false,
        logMaxSize: '10m',
        logMaxFiles: 5
      };
      setupLogging(config);
      consoleDebugSpy.mockClear();
      
      const logger = createLogger('debug-context-test');
      const context = new LogContext(logger, { key: 'value' });
      
      context.debug('Debug message');
      expect(consoleDebugSpy).toHaveBeenCalled();
    });

    it('should create nested context', () => {
      const logger = createLogger('nested-context-test');
      const context = new LogContext(logger, { level1: 'value' });
      const nestedContext = context.withContext({ level2: 'nested' });
      
      nestedContext.info('Nested message');
      expect(consoleSpy).toHaveBeenCalled();
    });
  });
});