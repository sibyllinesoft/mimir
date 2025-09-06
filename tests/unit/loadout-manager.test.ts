/**
 * Unit tests for LoadoutManager - Research loadout configuration management
 * 
 * Tests loadout loading, validation, caching, hot reload, and management operations.
 * Focuses on maximizing coverage with efficient, high-impact test cases.
 */

import { describe, expect, it, beforeEach, afterEach, spyOn, mock } from 'bun:test';
import { 
  LoadoutManager, 
  type LoadoutManagerConfig,
  type LoadoutInfo 
} from '@/research/loadout-manager';
import type { ResearchLoadout } from '@/research/types';

// Mock data
const mockValidLoadout: ResearchLoadout = {
  name: 'test-loadout',
  description: 'Test loadout for unit tests',
  version: '1.0.0',
  pipeline: {
    maxIterations: 3,
    convergenceThreshold: 0.8,
    parallelismLevel: 2,
    timeoutMs: 30000,
    focus: 'intelligence_gathering',
    outputOptimization: 'swarm_handoff'
  },
  agents: [
    {
      type: 'intelligence_analyst',
      name: 'primary-analyst',
      enabled: true,
      weight: 0.8,
      config: {
        systemPrompt: 'Test prompt',
        temperature: 0.7,
        maxTokens: 2000,
        model: 'gpt-4'
      },
      dependencies: []
    }
  ],
  stages: [
    {
      name: 'initial-analysis',
      type: 'domain_analysis',
      enabled: true,
      parallel: false,
      weight: 1.0,
      config: {
        depth: 'medium'
      }
    }
  ],
  verification: [
    {
      type: 'syntactic',
      enabled: true,
      weight: 0.5,
      config: {}
    }
  ],
  output: {
    format: 'intelligence_brief',
    sections: ['overview', 'analysis', 'recommendations'],
    densityTarget: 2000,
    optimizeFor: 'swarm_consumption',
    includeDebateTopics: true,
    includeMetrics: true,
    includeCitations: false
  },
  performance: {
    cacheResults: true,
    cacheTtlMinutes: 30,
    maxMemoryMb: 512,
    cpuIntensive: false
  }
};

const mockInvalidLoadout = {
  name: 'invalid-loadout',
  description: 'Missing required fields',
  version: '1.0.0'
  // Missing pipeline, agents, stages, verification, output, performance
};

const mockLoadoutYaml = `
name: yaml-loadout
description: Test YAML loadout
version: 1.1.0
pipeline:
  maxIterations: 5
  convergenceThreshold: 0.9
  parallelismLevel: 3
  timeoutMs: 45000
  focus: intelligence_gathering
  outputOptimization: swarm_handoff
agents:
  - type: strategy_mapper
    name: yaml-agent
    enabled: true
    weight: 0.7
    config:
      systemPrompt: YAML test prompt
      temperature: 0.5
    dependencies: []
stages:
  - name: yaml-stage
    type: technical_analysis
    enabled: true
    parallel: true
    weight: 0.8
    config:
      complexity: high
verification:
  - type: semantic
    enabled: true
    weight: 0.6
    config: {}
output:
  format: comprehensive
  sections: [overview, details]
  densityTarget: 3000
  optimizeFor: human_readable
  includeDebateTopics: false
  includeMetrics: true
  includeCitations: true
performance:
  cacheResults: false
  cacheTtlMinutes: 15
  maxMemoryMb: 256
  cpuIntensive: true
`;

const mockConfig: LoadoutManagerConfig = {
  loadoutDirectories: ['/test/loadouts'],
  enableHotReload: false,
  cacheEnabled: true,
  validationStrict: true,
  defaultLoadout: 'test-loadout'
};

// Mocks
const mockFs = {
  readdir: mock(),
  readFile: mock(),
  stat: mock(),
};

const mockWatch = mock();
const mockYaml = {
  parse: mock()
};

const mockLogger = {
  info: mock(),
  warn: mock(),
  error: mock(),
  debug: mock(),
};

// Mock the modules
mock.module('fs', () => ({
  promises: mockFs,
  watch: mockWatch,
}));

mock.module('yaml', () => ({
  parse: mockYaml.parse,
}));

mock.module('@/utils/logger', () => ({
  createLogger: mock(() => mockLogger),
}));

// Mock the validateLoadout function
mock.module('@/research/types', () => ({
  validateLoadout: mock((data: any) => {
    if (!data.pipeline || !data.agents || !data.stages) {
      throw new Error('Missing required fields');
    }
    return data;
  }),
  isValidLoadout: mock(() => true),
}));

describe('LoadoutManager', () => {
  let loadoutManager: LoadoutManager;

  beforeEach(() => {
    // Reset all mocks
    mockFs.readdir.mockReset();
    mockFs.readFile.mockReset();
    mockFs.stat.mockReset();
    mockWatch.mockReset();
    mockYaml.parse.mockReset();
    mockLogger.info.mockReset();
    mockLogger.warn.mockReset();
    mockLogger.error.mockReset();
    mockLogger.debug.mockReset();

    // Create fresh instance
    loadoutManager = new LoadoutManager(mockConfig);
  });

  describe('Constructor', () => {
    it('should create LoadoutManager with valid config', () => {
      expect(loadoutManager).toBeInstanceOf(LoadoutManager);
    });

    it('should create LoadoutManager with different configs', () => {
      const configs = [
        {
          ...mockConfig,
          enableHotReload: true,
          validationStrict: false
        },
        {
          ...mockConfig,
          loadoutDirectories: ['/dir1', '/dir2', '/dir3'],
          cacheEnabled: false
        },
        {
          ...mockConfig,
          defaultLoadout: ''
        }
      ];

      configs.forEach(config => {
        const manager = new LoadoutManager(config);
        expect(manager).toBeInstanceOf(LoadoutManager);
      });
    });
  });

  describe('Initialization', () => {
    it('should initialize without hot reload', async () => {
      mockFs.readdir.mockResolvedValue(['test.yaml', 'other.json']);
      mockFs.readFile.mockResolvedValue(JSON.stringify(mockValidLoadout));
      mockFs.stat.mockResolvedValue({ mtime: new Date() });

      await loadoutManager.initialize();
      
      expect(mockFs.readdir).toHaveBeenCalledWith('/test/loadouts');
      expect(mockWatch).not.toHaveBeenCalled();
    });

    it('should initialize with hot reload enabled', async () => {
      const config = { ...mockConfig, enableHotReload: true };
      const manager = new LoadoutManager(config);
      
      mockFs.readdir.mockResolvedValue([]);
      mockWatch.mockReturnValue({ close: mock() });

      await manager.initialize();
      
      expect(mockWatch).toHaveBeenCalled();
    });

    it('should handle initialization errors gracefully', async () => {
      mockFs.readdir.mockRejectedValue(new Error('Directory not found'));
      
      await expect(loadoutManager.initialize()).resolves.toBeUndefined();
    });
  });

  describe('Loading Loadouts from Directory', () => {
    it('should load YAML and JSON files from directory', async () => {
      const files = ['loadout1.json', 'loadout2.yaml', 'loadout3.yml', 'readme.txt', '.DS_Store'];
      mockFs.readdir.mockResolvedValue(files);
      mockFs.readFile.mockResolvedValueOnce(JSON.stringify(mockValidLoadout));
      mockFs.readFile.mockResolvedValueOnce(mockLoadoutYaml);
      mockFs.readFile.mockResolvedValueOnce(JSON.stringify({...mockValidLoadout, name: 'loadout3'}));
      mockFs.stat.mockResolvedValue({ mtime: new Date('2024-01-15') });
      mockYaml.parse.mockReturnValue({...mockValidLoadout, name: 'yaml-loadout'});

      await loadoutManager.initialize();

      expect(mockFs.readdir).toHaveBeenCalledWith('/test/loadouts');
      expect(mockFs.readFile).toHaveBeenCalledTimes(3); // Only YAML/JSON files
      expect(mockYaml.parse).toHaveBeenCalled();
    });

    it('should handle empty directories', async () => {
      mockFs.readdir.mockResolvedValue([]);
      
      await loadoutManager.initialize();
      
      expect(mockFs.readFile).not.toHaveBeenCalled();
    });

    it('should handle file read errors', async () => {
      mockFs.readdir.mockResolvedValue(['broken.json']);
      mockFs.readFile.mockRejectedValue(new Error('Permission denied'));
      mockFs.stat.mockResolvedValue({ mtime: new Date() });
      
      await loadoutManager.initialize();
      
      // Should not throw, just log error and continue
      expect(loadoutManager.getAllLoadouts().size).toBe(0);
    });
  });

  describe('Individual Loadout Loading', () => {
    it('should load valid JSON loadout', async () => {
      mockFs.readdir.mockResolvedValue(['valid.json']);
      mockFs.readFile.mockResolvedValue(JSON.stringify(mockValidLoadout));
      mockFs.stat.mockResolvedValue({ mtime: new Date('2024-01-20') });
      
      await loadoutManager.initialize();
      
      const loadout = loadoutManager.getLoadout('test-loadout');
      expect(loadout).toEqual(mockValidLoadout);
      expect(loadoutManager.isValidLoadout('test-loadout')).toBe(true);
    });

    it('should load valid YAML loadout', async () => {
      const yamlLoadout = {...mockValidLoadout, name: 'yaml-loadout'};
      mockFs.readdir.mockResolvedValue(['loadout.yaml']);
      mockFs.readFile.mockResolvedValue(mockLoadoutYaml);
      mockFs.stat.mockResolvedValue({ mtime: new Date('2024-01-21') });
      mockYaml.parse.mockReturnValue(yamlLoadout);
      
      await loadoutManager.initialize();
      
      const loadout = loadoutManager.getLoadout('yaml-loadout');
      expect(loadout).toEqual(yamlLoadout);
    });

    it('should handle JSON parse errors', async () => {
      mockFs.readdir.mockResolvedValue(['broken.json']);
      mockFs.readFile.mockResolvedValue('{ invalid json');
      mockFs.stat.mockResolvedValue({ mtime: new Date() });
      
      await loadoutManager.initialize();
      
      expect(loadoutManager.getAllLoadouts().size).toBe(0);
    });

    it('should handle YAML parse errors', async () => {
      mockFs.readdir.mockResolvedValue(['broken.yaml']);
      mockFs.readFile.mockResolvedValue('invalid: yaml: content:');
      mockFs.stat.mockResolvedValue({ mtime: new Date() });
      mockYaml.parse.mockImplementation(() => { throw new Error('Invalid YAML'); });
      
      await loadoutManager.initialize();
      
      expect(loadoutManager.getAllLoadouts().size).toBe(0);
    });

    it('should handle validation errors in strict mode', async () => {
      const strictConfig = { ...mockConfig, validationStrict: true };
      const manager = new LoadoutManager(strictConfig);
      
      mockFs.readdir.mockResolvedValue(['invalid.json']);
      mockFs.readFile.mockResolvedValue(JSON.stringify(mockInvalidLoadout));
      mockFs.stat.mockResolvedValue({ mtime: new Date() });
      
      await manager.initialize();
      
      expect(manager.getAllLoadouts().size).toBe(0);
    });

    it('should handle validation errors in non-strict mode', async () => {
      const nonStrictConfig = { ...mockConfig, validationStrict: false };
      const manager = new LoadoutManager(nonStrictConfig);
      
      mockFs.readdir.mockResolvedValue(['invalid.json']);
      mockFs.readFile.mockResolvedValue(JSON.stringify(mockInvalidLoadout));
      mockFs.stat.mockResolvedValue({ mtime: new Date() });
      
      await manager.initialize();
      
      const loadout = manager.getLoadout('invalid-loadout');
      expect(loadout).toEqual(mockInvalidLoadout);
      expect(manager.isValidLoadout('invalid-loadout')).toBe(false);
    });

    it('should use filename as loadout name if name missing', async () => {
      const noNameLoadout = {...mockValidLoadout};
      delete (noNameLoadout as any).name;
      
      mockFs.readdir.mockResolvedValue(['unnamed.json']);
      mockFs.readFile.mockResolvedValue(JSON.stringify(noNameLoadout));
      mockFs.stat.mockResolvedValue({ mtime: new Date() });
      
      await loadoutManager.initialize();
      
      const loadout = loadoutManager.getLoadout('unnamed');
      expect(loadout).toBeDefined();
    });
  });

  describe('Loadout Retrieval', () => {
    beforeEach(async () => {
      // Setup with multiple loadouts
      mockFs.readdir.mockResolvedValue(['loadout1.json', 'loadout2.yaml', 'invalid.json']);
      mockFs.readFile.mockResolvedValueOnce(JSON.stringify(mockValidLoadout));
      mockFs.readFile.mockResolvedValueOnce(mockLoadoutYaml);
      mockFs.readFile.mockResolvedValueOnce(JSON.stringify(mockInvalidLoadout));
      mockFs.stat.mockResolvedValue({ mtime: new Date() });
      mockYaml.parse.mockReturnValue({...mockValidLoadout, name: 'yaml-loadout'});
      
      const nonStrictConfig = { ...mockConfig, validationStrict: false };
      loadoutManager = new LoadoutManager(nonStrictConfig);
      await loadoutManager.initialize();
    });

    it('should get loadout by name', () => {
      const loadout = loadoutManager.getLoadout('test-loadout');
      expect(loadout?.name).toBe('test-loadout');
      
      const notFound = loadoutManager.getLoadout('non-existent');
      expect(notFound).toBeNull();
    });

    it('should get all loadouts', () => {
      const allLoadouts = loadoutManager.getAllLoadouts();
      expect(allLoadouts.size).toBe(3);
      expect(allLoadouts.has('test-loadout')).toBe(true);
      expect(allLoadouts.has('yaml-loadout')).toBe(true);
      expect(allLoadouts.has('invalid-loadout')).toBe(true);
    });

    it('should get only valid loadouts', () => {
      const validLoadouts = loadoutManager.getValidLoadouts();
      expect(validLoadouts).toHaveLength(2);
      expect(validLoadouts.some(l => l.name === 'test-loadout')).toBe(true);
      expect(validLoadouts.some(l => l.name === 'yaml-loadout')).toBe(true);
      expect(validLoadouts.some(l => l.name === 'invalid-loadout')).toBe(false);
    });

    it('should get loadout names', () => {
      const names = loadoutManager.getLoadoutNames();
      expect(names).toContain('test-loadout');
      expect(names).toContain('yaml-loadout');
      expect(names).toContain('invalid-loadout');
      expect(names).toHaveLength(3);
    });

    it('should get valid loadout names', () => {
      const validNames = loadoutManager.getValidLoadoutNames();
      expect(validNames).toContain('test-loadout');
      expect(validNames).toContain('yaml-loadout');
      expect(validNames).not.toContain('invalid-loadout');
      expect(validNames).toHaveLength(2);
    });

    it('should check if loadout exists', () => {
      expect(loadoutManager.hasLoadout('test-loadout')).toBe(true);
      expect(loadoutManager.hasLoadout('non-existent')).toBe(false);
    });

    it('should check if loadout is valid', () => {
      expect(loadoutManager.isValidLoadout('test-loadout')).toBe(true);
      expect(loadoutManager.isValidLoadout('invalid-loadout')).toBe(false);
      expect(loadoutManager.isValidLoadout('non-existent')).toBe(false);
    });
  });

  describe('Loadout Info and Metadata', () => {
    beforeEach(async () => {
      mockFs.readdir.mockResolvedValue(['test.json']);
      mockFs.readFile.mockResolvedValue(JSON.stringify(mockValidLoadout));
      mockFs.stat.mockResolvedValue({ mtime: new Date('2024-01-25T10:00:00Z') });
      
      await loadoutManager.initialize();
    });

    it('should get loadout info', () => {
      const info = loadoutManager.getLoadoutInfo('test-loadout');
      
      expect(info).toBeDefined();
      expect(info?.name).toBe('test-loadout');
      expect(info?.version).toBe('1.0.0');
      expect(info?.isValid).toBe(true);
      expect(info?.filePath).toContain('test.json');
      expect(info?.lastModified).toEqual(new Date('2024-01-25T10:00:00Z'));
      expect(info?.loadedAt).toBeInstanceOf(Date);
    });

    it('should get all loadout info', () => {
      const allInfo = loadoutManager.getAllLoadoutInfo();
      expect(allInfo).toHaveLength(1);
      expect(allInfo[0].name).toBe('test-loadout');
    });

    it('should return null for non-existent loadout info', () => {
      const info = loadoutManager.getLoadoutInfo('non-existent');
      expect(info).toBeNull();
    });

    it('should get validation errors', () => {
      // Add invalid loadout
      const nonStrictConfig = { ...mockConfig, validationStrict: false };
      const manager = new LoadoutManager(nonStrictConfig);
      
      const errors = manager.getLoadoutValidationErrors('test-loadout');
      expect(errors).toEqual([]);
      
      const nonExistentErrors = manager.getLoadoutValidationErrors('non-existent');
      expect(nonExistentErrors).toEqual([]);
    });
  });

  describe('Default Loadout Handling', () => {
    it('should return configured default loadout', async () => {
      mockFs.readdir.mockResolvedValue(['test.json']);
      mockFs.readFile.mockResolvedValue(JSON.stringify(mockValidLoadout));
      mockFs.stat.mockResolvedValue({ mtime: new Date() });
      
      await loadoutManager.initialize();
      
      const defaultLoadout = loadoutManager.getDefaultLoadout();
      expect(defaultLoadout?.name).toBe('test-loadout');
    });

    it('should fallback to first valid loadout if default not found', async () => {
      const config = { ...mockConfig, defaultLoadout: 'non-existent' };
      const manager = new LoadoutManager(config);
      
      mockFs.readdir.mockResolvedValue(['fallback.json']);
      mockFs.readFile.mockResolvedValue(JSON.stringify({...mockValidLoadout, name: 'fallback'}));
      mockFs.stat.mockResolvedValue({ mtime: new Date() });
      
      await manager.initialize();
      
      const defaultLoadout = manager.getDefaultLoadout();
      expect(defaultLoadout?.name).toBe('fallback');
    });

    it('should return null if no valid loadouts exist', async () => {
      mockFs.readdir.mockResolvedValue([]);
      
      await loadoutManager.initialize();
      
      const defaultLoadout = loadoutManager.getDefaultLoadout();
      expect(defaultLoadout).toBeNull();
    });
  });

  describe('Hot Reload Functionality', () => {

    it('should setup hot reload for directories', async () => {
      const config = { ...mockConfig, enableHotReload: true };
      const manager = new LoadoutManager(config);
      
      mockFs.readdir.mockResolvedValue([]);
      mockWatch.mockReturnValue({ close: mock() });
      
      await manager.initialize();
      
      expect(mockWatch).toHaveBeenCalled();
    });

    it('should handle hot reload setup errors', async () => {
      const config = { ...mockConfig, enableHotReload: true };
      const manager = new LoadoutManager(config);
      
      mockFs.readdir.mockResolvedValue([]);
      mockWatch.mockImplementation(() => {
        throw new Error('Watch setup failed');
      });
      
      // Should not throw
      await expect(manager.initialize()).resolves.toBeUndefined();
    });
  });

  describe('Dynamic Loadout Management', () => {
    beforeEach(async () => {
      mockFs.readdir.mockResolvedValue([]);
      await loadoutManager.initialize();
    });

    it('should add valid temporary loadout', () => {
      const result = loadoutManager.addLoadout(mockValidLoadout, true);
      
      expect(result).toBe(true);
      expect(loadoutManager.hasLoadout('test-loadout')).toBe(true);
      expect(loadoutManager.getLoadoutInfo('test-loadout')).toBeNull(); // Temporary, no info
    });

    it('should add valid permanent loadout', () => {
      const result = loadoutManager.addLoadout(mockValidLoadout, false);
      
      expect(result).toBe(true);
      expect(loadoutManager.hasLoadout('test-loadout')).toBe(true);
      
      const info = loadoutManager.getLoadoutInfo('test-loadout');
      expect(info).toBeDefined();
      expect(info?.filePath).toBe('<dynamic>');
    });

    it('should reject invalid loadout', () => {
      const result = loadoutManager.addLoadout(mockInvalidLoadout as any, false);
      
      expect(result).toBe(false);
      expect(loadoutManager.hasLoadout('invalid-loadout')).toBe(false);
    });

    it('should remove existing loadout', () => {
      loadoutManager.addLoadout(mockValidLoadout, false);
      
      const result = loadoutManager.removeLoadout('test-loadout');
      
      expect(result).toBe(true);
      expect(loadoutManager.hasLoadout('test-loadout')).toBe(false);
    });

    it('should return false when removing non-existent loadout', () => {
      const result = loadoutManager.removeLoadout('non-existent');
      
      expect(result).toBe(false);
    });
  });

  describe('Loadout Reloading', () => {
    beforeEach(async () => {
      mockFs.readdir.mockResolvedValue(['test.json']);
      mockFs.readFile.mockResolvedValue(JSON.stringify(mockValidLoadout));
      mockFs.stat.mockResolvedValue({ mtime: new Date() });
      
      await loadoutManager.initialize();
    });

    it('should reload existing loadout', async () => {
      const updatedLoadout = {...mockValidLoadout, version: '2.0.0'};
      mockFs.readFile.mockResolvedValueOnce(JSON.stringify(updatedLoadout));
      mockFs.stat.mockResolvedValueOnce({ mtime: new Date() });
      
      const result = await loadoutManager.reloadLoadout('test-loadout');
      
      expect(result).toBe(true);
      expect(loadoutManager.getLoadout('test-loadout')?.version).toBe('2.0.0');
    });

    it('should return false for unknown loadout', async () => {
      const result = await loadoutManager.reloadLoadout('unknown');
      
      expect(result).toBe(false);
    });

    it('should handle reload errors', async () => {
      mockFs.readFile.mockRejectedValueOnce(new Error('File not found'));
      
      const result = await loadoutManager.reloadLoadout('test-loadout');
      
      expect(result).toBe(false);
    });

    it('should reload all loadouts', async () => {
      mockFs.readdir.mockResolvedValueOnce(['updated.json']);
      mockFs.readFile.mockResolvedValueOnce(JSON.stringify({...mockValidLoadout, name: 'updated'}));
      mockFs.stat.mockResolvedValueOnce({ mtime: new Date() });
      
      await loadoutManager.reloadAllLoadouts();
      
      expect(loadoutManager.hasLoadout('test-loadout')).toBe(false);
      expect(loadoutManager.hasLoadout('updated')).toBe(true);
    });
  });

  describe('Validation', () => {
    beforeEach(async () => {
      const nonStrictConfig = { ...mockConfig, validationStrict: false };
      loadoutManager = new LoadoutManager(nonStrictConfig);
      
      mockFs.readdir.mockResolvedValue(['valid.json', 'invalid.json']);
      mockFs.readFile.mockResolvedValueOnce(JSON.stringify(mockValidLoadout));
      mockFs.readFile.mockResolvedValueOnce(JSON.stringify(mockInvalidLoadout));
      mockFs.stat.mockResolvedValue({ mtime: new Date() });
      
      await loadoutManager.initialize();
    });

    it('should validate loadout by name - valid', () => {
      const result = loadoutManager.validateLoadoutByName('test-loadout');
      
      expect(result.valid).toBe(true);
      expect(result.errors).toEqual([]);
    });

    it('should validate loadout by name - invalid', () => {
      const result = loadoutManager.validateLoadoutByName('invalid-loadout');
      
      expect(result.valid).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);
    });

    it('should handle validation for non-existent loadout', () => {
      const result = loadoutManager.validateLoadoutByName('non-existent');
      
      expect(result.valid).toBe(false);
      expect(result.errors).toEqual(['Loadout not found']);
    });
  });

  describe('Statistics', () => {
    beforeEach(async () => {
      const nonStrictConfig = { ...mockConfig, validationStrict: false };
      loadoutManager = new LoadoutManager(nonStrictConfig);
      
      mockFs.readdir.mockResolvedValue(['valid.json', 'invalid.json']);
      mockFs.readFile.mockResolvedValueOnce(JSON.stringify(mockValidLoadout));
      mockFs.readFile.mockResolvedValueOnce(JSON.stringify(mockInvalidLoadout));
      mockFs.stat.mockResolvedValue({ mtime: new Date() });
      
      await loadoutManager.initialize();
    });

    it('should provide comprehensive statistics', () => {
      const stats = loadoutManager.getStatistics();
      
      expect(stats).toEqual({
        totalLoadouts: 2,
        validLoadouts: 1,
        invalidLoadouts: 1,
        directories: 1,
        hotReloadEnabled: false,
        cacheEnabled: true,
        averageAgentsPerLoadout: 1,
        averageStagesPerLoadout: 1
      });
    });

    it('should handle empty statistics', () => {
      const emptyManager = new LoadoutManager(mockConfig);
      const stats = emptyManager.getStatistics();
      
      expect(stats.totalLoadouts).toBe(0);
      expect(stats.validLoadouts).toBe(0);
      expect(stats.averageAgentsPerLoadout).toBe(0);
      expect(stats.averageStagesPerLoadout).toBe(0);
    });
  });

  describe('Cleanup', () => {
    it('should cleanup watchers and clear caches', async () => {
      const config = { ...mockConfig, enableHotReload: true };
      const manager = new LoadoutManager(config);
      
      const mockWatcher = { close: mock() };
      mockWatch.mockReturnValue(mockWatcher);
      mockFs.readdir.mockResolvedValue([]);
      
      await manager.initialize();
      await manager.cleanup();
      
      expect(mockWatcher.close).toHaveBeenCalled();
    });

    it('should handle watcher cleanup errors', async () => {
      const config = { ...mockConfig, enableHotReload: true };
      const manager = new LoadoutManager(config);
      
      const mockWatcher = { 
        close: mock(() => {
          throw new Error('Close failed');
        })
      };
      mockWatch.mockReturnValue(mockWatcher);
      mockFs.readdir.mockResolvedValue([]);
      
      await manager.initialize();
      
      // Should not throw
      await expect(manager.cleanup()).resolves.toBeUndefined();
    });
  });

  describe('Multiple Directories', () => {
    it('should load from multiple directories', async () => {
      const multiDirConfig = {
        ...mockConfig,
        loadoutDirectories: ['/dir1', '/dir2']
      };
      const manager = new LoadoutManager(multiDirConfig);
      
      mockFs.readdir.mockResolvedValueOnce(['loadout1.json']);
      mockFs.readdir.mockResolvedValueOnce(['loadout2.yaml']);
      mockFs.readFile.mockResolvedValueOnce(JSON.stringify(mockValidLoadout));
      mockFs.readFile.mockResolvedValueOnce(mockLoadoutYaml);
      mockFs.stat.mockResolvedValue({ mtime: new Date() });
      mockYaml.parse.mockReturnValue({...mockValidLoadout, name: 'yaml-loadout'});
      
      await manager.initialize();
      
      expect(mockFs.readdir).toHaveBeenCalledWith('/dir1');
      expect(mockFs.readdir).toHaveBeenCalledWith('/dir2');
      expect(manager.getAllLoadouts().size).toBe(2);
    });

    it('should continue loading despite directory errors', async () => {
      const multiDirConfig = {
        ...mockConfig,
        loadoutDirectories: ['/good-dir', '/bad-dir']
      };
      const manager = new LoadoutManager(multiDirConfig);
      
      mockFs.readdir.mockResolvedValueOnce(['good.json']);
      mockFs.readdir.mockRejectedValueOnce(new Error('Permission denied'));
      mockFs.readFile.mockResolvedValue(JSON.stringify(mockValidLoadout));
      mockFs.stat.mockResolvedValue({ mtime: new Date() });
      
      await manager.initialize();
      
      expect(manager.getAllLoadouts().size).toBe(1);
    });
  });

  describe('Hot Reload File Watching', () => {
    it('should handle file change events', async () => {
      const config = { ...mockConfig, enableHotReload: true };
      const manager = new LoadoutManager(config);
      
      let watchCallback: any;
      mockWatch.mockImplementation((dir, options, callback) => {
        watchCallback = callback;
        return { close: mock() };
      });
      
      mockFs.readdir.mockResolvedValue(['existing.json']);
      mockFs.readFile.mockResolvedValueOnce(JSON.stringify(mockValidLoadout));
      mockFs.stat.mockResolvedValue({ mtime: new Date() });
      
      await manager.initialize();
      
      // Set up for file change
      const updatedLoadout = { ...mockValidLoadout, version: '2.0.0' };
      mockFs.readFile.mockResolvedValueOnce(JSON.stringify(updatedLoadout));
      
      // Simulate file change
      await watchCallback('change', 'existing.json');
      
      // Should handle the change without throwing
      expect(mockFs.readFile).toHaveBeenCalledTimes(2);
    });

    it('should handle file rename events', async () => {
      const config = { ...mockConfig, enableHotReload: true };
      const manager = new LoadoutManager(config);
      
      let watchCallback: any;
      mockWatch.mockImplementation((dir, options, callback) => {
        watchCallback = callback;
        return { close: mock() };
      });
      
      mockFs.readdir.mockResolvedValue(['existing.json']);
      mockFs.readFile.mockResolvedValueOnce(JSON.stringify(mockValidLoadout));
      mockFs.stat.mockResolvedValue({ mtime: new Date() });
      
      await manager.initialize();
      
      // Set up for file rename
      mockFs.readFile.mockResolvedValueOnce(JSON.stringify(mockValidLoadout));
      
      // Simulate file rename
      await watchCallback('rename', 'existing.json');
      
      // Should handle the rename without throwing
      expect(mockFs.readFile).toHaveBeenCalledTimes(2);
    });

    it('should ignore non-loadout files in watch events', async () => {
      const config = { ...mockConfig, enableHotReload: true };
      const manager = new LoadoutManager(config);
      
      let watchCallback: any;
      mockWatch.mockImplementation((dir, options, callback) => {
        watchCallback = callback;
        return { close: mock() };
      });
      
      mockFs.readdir.mockResolvedValue([]);
      
      await manager.initialize();
      
      const readFileCalls = mockFs.readFile.mock.calls.length;
      
      // Simulate non-loadout file change
      await watchCallback('change', 'readme.txt');
      
      // Should not call readFile again
      expect(mockFs.readFile.mock.calls.length).toBe(readFileCalls);
    });

    it('should handle watch callback with no filename', async () => {
      const config = { ...mockConfig, enableHotReload: true };
      const manager = new LoadoutManager(config);
      
      let watchCallback: any;
      mockWatch.mockImplementation((dir, options, callback) => {
        watchCallback = callback;
        return { close: mock() };
      });
      
      mockFs.readdir.mockResolvedValue([]);
      
      await manager.initialize();
      
      const readFileCalls = mockFs.readFile.mock.calls.length;
      
      // Simulate watch event with no filename
      await watchCallback('change', null);
      
      // Should not call readFile again
      expect(mockFs.readFile.mock.calls.length).toBe(readFileCalls);
    });
  });

  describe('Loadout Comparison', () => {
    beforeEach(async () => {
      const nonStrictConfig = { ...mockConfig, validationStrict: false };
      loadoutManager = new LoadoutManager(nonStrictConfig);
      
      // Set up multiple loadouts with different characteristics
      const fastLoadout = {
        ...mockValidLoadout,
        name: 'fast-loadout',
        pipeline: { ...mockValidLoadout.pipeline, parallelismLevel: 8, maxIterations: 1 }
      };
      
      const thoroughLoadout = {
        ...mockValidLoadout,
        name: 'thorough-loadout',
        pipeline: { ...mockValidLoadout.pipeline, maxIterations: 6, parallelismLevel: 2 },
        output: { ...mockValidLoadout.output, densityTarget: 5000 }
      };

      mockFs.readdir.mockResolvedValue(['test.json', 'fast.json', 'thorough.json']);
      mockFs.readFile.mockResolvedValueOnce(JSON.stringify(mockValidLoadout));
      mockFs.readFile.mockResolvedValueOnce(JSON.stringify(fastLoadout));
      mockFs.readFile.mockResolvedValueOnce(JSON.stringify(thoroughLoadout));
      mockFs.stat.mockResolvedValue({ mtime: new Date() });
      
      await loadoutManager.initialize();
    });

    it('should compare multiple loadouts', () => {
      const comparison = loadoutManager.compareLoadouts(['test-loadout', 'fast-loadout', 'thorough-loadout']);
      
      expect(comparison).toBeDefined();
      expect(comparison.loadouts).toEqual(['test-loadout', 'fast-loadout', 'thorough-loadout']);
      expect(comparison.commonFeatures).toBeInstanceOf(Array);
      expect(comparison.differences).toBeInstanceOf(Array);
      expect(comparison.recommendations).toBeInstanceOf(Array);
    });

    it('should find common features across loadouts', () => {
      const comparison = loadoutManager.compareLoadouts(['test-loadout', 'fast-loadout']);
      
      expect(comparison.commonFeatures.length).toBeGreaterThan(0);
      expect(comparison.commonFeatures.some(f => f.includes('Common agents'))).toBe(true);
    });

    it('should identify differences between loadouts', () => {
      const comparison = loadoutManager.compareLoadouts(['fast-loadout', 'thorough-loadout']);
      
      expect(comparison.differences.length).toBeGreaterThan(0);
      expect(comparison.differences.some(d => d.includes('max iterations'))).toBe(true);
      expect(comparison.differences.some(d => d.includes('parallelism levels'))).toBe(true);
    });

    it('should generate performance recommendations', () => {
      const comparison = loadoutManager.compareLoadouts(['fast-loadout', 'thorough-loadout']);
      
      expect(comparison.recommendations.length).toBeGreaterThan(0);
      expect(comparison.recommendations.some(r => r.includes('speed'))).toBe(true);
      expect(comparison.recommendations.some(r => r.includes('thoroughness'))).toBe(true);
    });

    it('should handle comparison with non-existent loadouts', () => {
      const comparison = loadoutManager.compareLoadouts(['test-loadout', 'non-existent']);
      
      expect(comparison.loadouts).toEqual(['test-loadout']);
      expect(comparison.commonFeatures).toBeDefined();
      expect(comparison.differences).toBeDefined();
      expect(comparison.recommendations).toBeDefined();
    });

    it('should handle empty comparison list', () => {
      const comparison = loadoutManager.compareLoadouts([]);
      
      expect(comparison.loadouts).toEqual([]);
      expect(comparison.commonFeatures).toEqual([]);
      expect(comparison.differences).toEqual([]);
      expect(comparison.recommendations).toEqual([]);
    });

    it('should find common output format', () => {
      const comparison = loadoutManager.compareLoadouts(['test-loadout', 'fast-loadout']);
      
      expect(comparison.commonFeatures.some(f => f.includes('Common output format'))).toBe(true);
    });
  });
});