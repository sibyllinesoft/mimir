/**
 * Logging utilities for structured event tracking
 * 
 * Provides structured logging with JSON event streams and console output
 * for pipeline progress monitoring.
 */

import type { Logger } from '@/types';
import type { LoggingConfig } from '@/config/config';

export enum LogLevel {
  DEBUG = 'debug',
  INFO = 'info',
  WARN = 'warn',
  ERROR = 'error',
}

interface LogEntry {
  timestamp: string;
  level: string;
  logger: string;
  message: string;
  context?: Record<string, any>;
  error?: string;
}

export class MimirLogger implements Logger {
  private name: string;
  private level: LogLevel;
  private format: 'json' | 'text' | 'human';
  private context: Record<string, any> = {};

  constructor(name: string, level: LogLevel = LogLevel.INFO, format: 'json' | 'text' | 'human' = 'json') {
    this.name = name;
    this.level = level;
    this.format = format;
  }

  private shouldLog(level: LogLevel): boolean {
    const levels = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARN, LogLevel.ERROR];
    const currentIndex = levels.indexOf(this.level);
    const messageIndex = levels.indexOf(level);
    return messageIndex >= currentIndex;
  }

  private formatMessage(level: LogLevel, message: string, meta?: any): string {
    const timestamp = new Date().toISOString();
    
    const entry: LogEntry = {
      timestamp,
      level,
      logger: this.name,
      message,
      ...(Object.keys(this.context).length > 0 && { context: this.context }),
      ...(meta && { ...meta }),
    };

    if (this.format === 'json') {
      return JSON.stringify(entry);
    } else if (this.format === 'human') {
      const levelEmojis = {
        [LogLevel.DEBUG]: '🔍',
        [LogLevel.INFO]: 'ℹ️',
        [LogLevel.WARN]: '⚠️',
        [LogLevel.ERROR]: '❌',
      };
      const emoji = levelEmojis[level];
      const time = new Date().toLocaleTimeString();
      return `${emoji} ${time} [${this.name}] ${message}`;
    } else {
      // text format
      return `${timestamp} - ${this.name} - ${level.toUpperCase()} - ${message}`;
    }
  }

  private write(level: LogLevel, message: string, meta?: any): void {
    if (!this.shouldLog(level)) return;
    
    const formattedMessage = this.formatMessage(level, message, meta);
    
    // Use appropriate console method
    switch (level) {
      case LogLevel.ERROR:
        console.error(formattedMessage);
        break;
      case LogLevel.WARN:
        console.warn(formattedMessage);
        break;
      case LogLevel.DEBUG:
        console.debug(formattedMessage);
        break;
      default:
        console.log(formattedMessage);
    }
  }

  debug(message: string, meta?: any): void {
    this.write(LogLevel.DEBUG, message, meta);
  }

  info(message: string, meta?: any): void {
    this.write(LogLevel.INFO, message, meta);
  }

  warn(message: string, meta?: any): void {
    this.write(LogLevel.WARN, message, meta);
  }

  error(message: string, meta?: any): void {
    this.write(LogLevel.ERROR, message, meta);
  }

  withContext(context: Record<string, any>): Logger {
    const childLogger = new MimirLogger(this.name, this.level, this.format);
    childLogger.context = { ...this.context, ...context };
    return childLogger;
  }

  child(name: string): Logger {
    const childName = `${this.name}.${name}`;
    const childLogger = new MimirLogger(childName, this.level, this.format);
    childLogger.context = { ...this.context };
    return childLogger;
  }
}

// Global logger management
const loggers = new Map<string, MimirLogger>();
let defaultLevel: LogLevel = LogLevel.INFO;
let defaultFormat: 'json' | 'text' | 'human' = 'json';

export function setupLogging(config?: LoggingConfig): void {
  if (config) {
    defaultLevel = config.logLevel as LogLevel;
    defaultFormat = config.logFormat;
  }

  // Set specific logger levels for common noisy libraries
  const silencedLoggers = ['mcp', 'httpx', 'fetch'];
  silencedLoggers.forEach(name => {
    const logger = createLogger(name);
    (logger as MimirLogger).level = LogLevel.WARN;
  });
}

export function createLogger(name: string): Logger {
  if (!loggers.has(name)) {
    loggers.set(name, new MimirLogger(name, defaultLevel, defaultFormat));
  }
  return loggers.get(name)!;
}

export function getLogger(name: string): Logger {
  return createLogger(name);
}

// Pipeline-specific logging utilities
export interface PipelineStage {
  name: string;
  progress?: number;
  duration?: number;
}

export class PipelineLogger {
  private logger: Logger;
  private indexId: string;
  private startTime: number;

  constructor(indexId: string, logger?: Logger) {
    this.indexId = indexId;
    this.logger = logger || createLogger(`mimir.pipeline.${indexId}`);
    this.startTime = Date.now();
  }

  logStageStart(stage: string, message?: string): void {
    const msg = message || `Starting ${stage} stage`;
    this.logger.info(msg, {
      indexId: this.indexId,
      stage,
      progress: 0,
      event: 'stage_start',
    });
  }

  logStageProgress(stage: string, progress: number, message?: string): void {
    const msg = message || `${stage} progress: ${progress}%`;
    this.logger.info(msg, {
      indexId: this.indexId,
      stage,
      progress,
      event: 'stage_progress',
    });
  }

  logStageComplete(stage: string, message?: string): void {
    const duration = Date.now() - this.startTime;
    const msg = message || `Completed ${stage} stage`;
    this.logger.info(msg, {
      indexId: this.indexId,
      stage,
      progress: 100,
      duration,
      event: 'stage_complete',
    });
  }

  logStageError(stage: string, error: Error, message?: string): void {
    const msg = message || `Error in ${stage} stage: ${error.message}`;
    this.logger.error(msg, {
      indexId: this.indexId,
      stage,
      error: error.message,
      errorType: error.constructor.name,
      stack: error.stack,
      event: 'stage_error',
    });
  }

  logInfo(message: string, stage?: string, meta?: any): void {
    this.logger.info(message, {
      indexId: this.indexId,
      ...(stage && { stage }),
      ...meta,
    });
  }

  logWarning(message: string, stage?: string, meta?: any): void {
    this.logger.warn(message, {
      indexId: this.indexId,
      ...(stage && { stage }),
      ...meta,
    });
  }

  logError(message: string, error?: Error, stage?: string, meta?: any): void {
    this.logger.error(message, {
      indexId: this.indexId,
      ...(stage && { stage }),
      ...(error && {
        error: error.message,
        errorType: error.constructor.name,
        stack: error.stack,
      }),
      ...meta,
    });
  }
}

export function getPipelineLogger(indexId: string): PipelineLogger {
  return new PipelineLogger(indexId);
}

// Context manager for structured logging
export class LogContext {
  private logger: Logger;
  private context: Record<string, any>;

  constructor(logger: Logger, context: Record<string, any>) {
    this.logger = logger;
    this.context = context;
  }

  debug(message: string, meta?: any): void {
    this.logger.debug(message, { ...this.context, ...meta });
  }

  info(message: string, meta?: any): void {
    this.logger.info(message, { ...this.context, ...meta });
  }

  warn(message: string, meta?: any): void {
    this.logger.warn(message, { ...this.context, ...meta });
  }

  error(message: string, meta?: any): void {
    this.logger.error(message, { ...this.context, ...meta });
  }

  withContext(additionalContext: Record<string, any>): LogContext {
    return new LogContext(this.logger, { ...this.context, ...additionalContext });
  }
}

export function withContext(logger: Logger, context: Record<string, any>): LogContext {
  return new LogContext(logger, context);
}
