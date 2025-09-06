/**
 * Loadout Configuration System
 * 
 * Manages loading, validation, and caching of research loadout configurations
 * from YAML/JSON files. Supports hot reloading and validation.
 */

import { promises as fs } from 'fs';
import { join, extname, basename } from 'path';
import * as yaml from 'yaml';
import { watch } from 'fs';
import type { Logger } from '@/types';
import { createLogger } from '@/utils/logger';
import { 
  ResearchLoadout, 
  ResearchLoadoutSchema, 
  validateLoadout,
  isValidLoadout 
} from './types';

export interface LoadoutManagerConfig {
  loadoutDirectories: string[];
  enableHotReload: boolean;
  cacheEnabled: boolean;
  validationStrict: boolean;
  defaultLoadout: string;
}

export interface LoadoutInfo {
  name: string;
  filePath: string;
  lastModified: Date;
  loadedAt: Date;
  version: string;
  isValid: boolean;
  validationErrors?: string[];
}

export class LoadoutManager {
  private loadouts = new Map<string, ResearchLoadout>();
  private loadoutInfo = new Map<string, LoadoutInfo>();
  private watchers = new Map<string, any>();
  private logger: Logger;

  constructor(private config: LoadoutManagerConfig) {
    this.logger = createLogger('mimir.research.loadout-manager');
  }

  async initialize(): Promise<void> {
    this.logger.info('Initializing loadout manager', { 
      directories: this.config.loadoutDirectories,
      hotReload: this.config.enableHotReload 
    });

    // Load all loadouts from configured directories
    await this.loadAllLoadouts();

    // Set up hot reloading if enabled
    if (this.config.enableHotReload) {
      await this.setupHotReloading();
    }

    this.logger.info('Loadout manager initialized', { 
      loadoutsLoaded: this.loadouts.size,
      validLoadouts: this.getValidLoadouts().length 
    });
  }

  async loadAllLoadouts(): Promise<void> {
    const loadPromises = this.config.loadoutDirectories.map(dir => 
      this.loadLoadoutsFromDirectory(dir)
    );

    await Promise.allSettled(loadPromises);
  }

  private async loadLoadoutsFromDirectory(directory: string): Promise<void> {
    try {
      const files = await fs.readdir(directory);
      const loadoutFiles = files.filter(file => 
        file.endsWith('.yml') || 
        file.endsWith('.yaml') || 
        file.endsWith('.json')
      );

      for (const file of loadoutFiles) {
        const filePath = join(directory, file);
        await this.loadLoadoutFromFile(filePath);
      }

      this.logger.info('Loaded loadouts from directory', { 
        directory, 
        filesProcessed: loadoutFiles.length 
      });
    } catch (error) {
      this.logger.error('Failed to load loadouts from directory', { 
        directory, 
        error 
      });
    }
  }

  private async loadLoadoutFromFile(filePath: string): Promise<LoadoutInfo | null> {
    try {
      const fileName = basename(filePath, extname(filePath));
      const content = await fs.readFile(filePath, 'utf-8');
      const stats = await fs.stat(filePath);

      let loadoutData: any;
      if (filePath.endsWith('.json')) {
        loadoutData = JSON.parse(content);
      } else {
        loadoutData = yaml.parse(content);
      }

      // Validate the loadout
      const validationErrors: string[] = [];
      let isValid = true;
      let validatedLoadout: ResearchLoadout;

      try {
        validatedLoadout = validateLoadout(loadoutData);
      } catch (error) {
        isValid = false;
        if (error instanceof Error) {
          validationErrors.push(error.message);
        } else {
          validationErrors.push('Unknown validation error');
        }
        
        if (this.config.validationStrict) {
          this.logger.error('Strict validation failed for loadout', { 
            filePath, 
            errors: validationErrors 
          });
          return null;
        }

        // In non-strict mode, try to use the data anyway
        validatedLoadout = loadoutData as ResearchLoadout;
      }

      // Store the loadout
      const loadoutName = validatedLoadout?.name || fileName;
      this.loadouts.set(loadoutName, validatedLoadout);

      // Store metadata
      const info: LoadoutInfo = {
        name: loadoutName,
        filePath,
        lastModified: stats.mtime,
        loadedAt: new Date(),
        version: validatedLoadout?.version || '0.0.0',
        isValid,
        validationErrors: validationErrors.length > 0 ? validationErrors : undefined,
      };

      this.loadoutInfo.set(loadoutName, info);

      this.logger.info('Loaded loadout', { 
        name: loadoutName, 
        filePath, 
        valid: isValid,
        errors: validationErrors.length 
      });

      return info;
    } catch (error) {
      this.logger.error('Failed to load loadout from file', { filePath, error });
      return null;
    }
  }

  private async setupHotReloading(): Promise<void> {
    for (const directory of this.config.loadoutDirectories) {
      try {
        const watcher = watch(directory, { recursive: false }, async (eventType, filename) => {
          if (!filename || (!filename.endsWith('.yml') && !filename.endsWith('.yaml') && !filename.endsWith('.json'))) {
            return;
          }

          const filePath = join(directory, filename);
          
          if (eventType === 'change' || eventType === 'rename') {
            this.logger.info('Loadout file changed, reloading', { filePath, eventType });
            
            // Remove old loadout if it exists
            const oldInfo = Array.from(this.loadoutInfo.values()).find(info => info.filePath === filePath);
            if (oldInfo) {
              this.loadouts.delete(oldInfo.name);
              this.loadoutInfo.delete(oldInfo.name);
            }

            // Try to load the new/changed loadout
            try {
              await this.loadLoadoutFromFile(filePath);
            } catch (error) {
              this.logger.error('Failed to reload loadout', { filePath, error });
            }
          }
        });

        this.watchers.set(directory, watcher);
        this.logger.info('Hot reloading enabled for directory', { directory });
      } catch (error) {
        this.logger.error('Failed to setup hot reloading for directory', { directory, error });
      }
    }
  }

  getLoadout(name: string): ResearchLoadout | null {
    return this.loadouts.get(name) || null;
  }

  getAllLoadouts(): Map<string, ResearchLoadout> {
    return new Map(this.loadouts);
  }

  getValidLoadouts(): ResearchLoadout[] {
    return Array.from(this.loadoutInfo.entries())
      .filter(([_, info]) => info.isValid)
      .map(([name, _]) => this.loadouts.get(name)!)
      .filter(Boolean);
  }

  getLoadoutInfo(name: string): LoadoutInfo | null {
    return this.loadoutInfo.get(name) || null;
  }

  getAllLoadoutInfo(): LoadoutInfo[] {
    return Array.from(this.loadoutInfo.values());
  }

  getLoadoutNames(): string[] {
    return Array.from(this.loadouts.keys());
  }

  getValidLoadoutNames(): string[] {
    return Array.from(this.loadoutInfo.entries())
      .filter(([_, info]) => info.isValid)
      .map(([name, _]) => name);
  }

  getDefaultLoadout(): ResearchLoadout | null {
    // Try to get the configured default loadout
    if (this.config.defaultLoadout && this.loadouts.has(this.config.defaultLoadout)) {
      return this.loadouts.get(this.config.defaultLoadout)!;
    }

    // Fallback to first valid loadout
    const validNames = this.getValidLoadoutNames();
    if (validNames.length > 0) {
      return this.loadouts.get(validNames[0])!;
    }

    return null;
  }

  hasLoadout(name: string): boolean {
    return this.loadouts.has(name);
  }

  isValidLoadout(name: string): boolean {
    const info = this.loadoutInfo.get(name);
    return info ? info.isValid : false;
  }

  getLoadoutValidationErrors(name: string): string[] {
    const info = this.loadoutInfo.get(name);
    return info?.validationErrors || [];
  }

  async reloadLoadout(name: string): Promise<boolean> {
    const info = this.loadoutInfo.get(name);
    if (!info) {
      this.logger.warn('Cannot reload unknown loadout', { name });
      return false;
    }

    try {
      // Remove the old loadout
      this.loadouts.delete(name);
      this.loadoutInfo.delete(name);

      // Reload from file
      const newInfo = await this.loadLoadoutFromFile(info.filePath);
      return newInfo !== null;
    } catch (error) {
      this.logger.error('Failed to reload loadout', { name, error });
      return false;
    }
  }

  async reloadAllLoadouts(): Promise<void> {
    this.loadouts.clear();
    this.loadoutInfo.clear();
    await this.loadAllLoadouts();
  }

  validateLoadoutByName(name: string): { valid: boolean; errors: string[] } {
    const loadout = this.loadouts.get(name);
    if (!loadout) {
      return { valid: false, errors: ['Loadout not found'] };
    }

    try {
      validateLoadout(loadout);
      return { valid: true, errors: [] };
    } catch (error) {
      const errors = error instanceof Error ? [error.message] : ['Unknown validation error'];
      return { valid: false, errors };
    }
  }

  // Dynamic loadout creation and management
  addLoadout(loadout: ResearchLoadout, temporary = true): boolean {
    try {
      validateLoadout(loadout);
      
      this.loadouts.set(loadout.name, loadout);
      
      if (!temporary) {
        // Add metadata for permanent loadouts
        const info: LoadoutInfo = {
          name: loadout.name,
          filePath: '<dynamic>',
          lastModified: new Date(),
          loadedAt: new Date(),
          version: loadout.version,
          isValid: true,
        };
        this.loadoutInfo.set(loadout.name, info);
      }

      this.logger.info('Added loadout', { 
        name: loadout.name, 
        temporary,
        version: loadout.version 
      });
      return true;
    } catch (error) {
      this.logger.error('Failed to add loadout', { 
        name: loadout.name, 
        error 
      });
      return false;
    }
  }

  removeLoadout(name: string): boolean {
    const existed = this.loadouts.has(name);
    this.loadouts.delete(name);
    this.loadoutInfo.delete(name);
    
    if (existed) {
      this.logger.info('Removed loadout', { name });
    }
    
    return existed;
  }

  // Loadout comparison utilities
  compareLoadouts(names: string[]): LoadoutComparison {
    const loadouts = names.map(name => ({
      name,
      loadout: this.loadouts.get(name),
      info: this.loadoutInfo.get(name)
    })).filter(item => item.loadout);

    const comparison: LoadoutComparison = {
      loadouts: loadouts.map(item => item.name),
      commonFeatures: this.findCommonFeatures(loadouts.map(item => item.loadout!)),
      differences: this.findDifferences(loadouts.map(item => item.loadout!)),
      recommendations: this.generateComparisonRecommendations(loadouts.map(item => item.loadout!)),
    };

    return comparison;
  }

  private findCommonFeatures(loadouts: ResearchLoadout[]): string[] {
    if (loadouts.length === 0) return [];
    
    const features: string[] = [];
    
    // Check common agent types
    const firstAgentTypes = new Set(loadouts[0].agents.filter(a => a.enabled).map(a => a.type));
    const commonAgentTypes = Array.from(firstAgentTypes).filter(type =>
      loadouts.every(loadout => loadout.agents.some(agent => agent.type === type && agent.enabled))
    );
    
    if (commonAgentTypes.length > 0) {
      features.push(`Common agents: ${commonAgentTypes.join(', ')}`);
    }

    // Check common stages
    const firstStageTypes = new Set(loadouts[0].stages.filter(s => s.enabled).map(s => s.type));
    const commonStageTypes = Array.from(firstStageTypes).filter(type =>
      loadouts.every(loadout => loadout.stages.some(stage => stage.type === type && stage.enabled))
    );

    if (commonStageTypes.length > 0) {
      features.push(`Common stages: ${commonStageTypes.join(', ')}`);
    }

    // Check common output format
    const outputFormats = loadouts.map(l => l.output.format);
    if (new Set(outputFormats).size === 1) {
      features.push(`Common output format: ${outputFormats[0]}`);
    }

    return features;
  }

  private findDifferences(loadouts: ResearchLoadout[]): string[] {
    const differences: string[] = [];
    
    // Compare pipeline settings
    const maxIterations = loadouts.map(l => l.pipeline.maxIterations);
    if (new Set(maxIterations).size > 1) {
      differences.push(`Different max iterations: ${maxIterations.join(', ')}`);
    }

    const parallelismLevels = loadouts.map(l => l.pipeline.parallelismLevel);
    if (new Set(parallelismLevels).size > 1) {
      differences.push(`Different parallelism levels: ${parallelismLevels.join(', ')}`);
    }

    // Compare density targets
    const densityTargets = loadouts.map(l => l.output.densityTarget);
    if (new Set(densityTargets).size > 1) {
      differences.push(`Different density targets: ${densityTargets.join(', ')}`);
    }

    return differences;
  }

  private generateComparisonRecommendations(loadouts: ResearchLoadout[]): string[] {
    const recommendations: string[] = [];

    // Performance vs quality recommendations
    const highPerformance = loadouts.filter(l => 
      l.pipeline.parallelismLevel >= 5 && 
      l.pipeline.maxIterations <= 2
    );

    const deepAnalysis = loadouts.filter(l => 
      l.pipeline.maxIterations >= 4 && 
      l.output.densityTarget >= 4000
    );

    if (highPerformance.length > 0) {
      recommendations.push(`For speed: ${highPerformance.map(l => l.name).join(', ')}`);
    }

    if (deepAnalysis.length > 0) {
      recommendations.push(`For thoroughness: ${deepAnalysis.map(l => l.name).join(', ')}`);
    }

    return recommendations;
  }

  async cleanup(): Promise<void> {
    // Close all file watchers
    for (const [directory, watcher] of this.watchers) {
      try {
        watcher.close();
        this.logger.info('Closed file watcher', { directory });
      } catch (error) {
        this.logger.error('Failed to close file watcher', { directory, error });
      }
    }

    this.watchers.clear();
    this.loadouts.clear();
    this.loadoutInfo.clear();
    
    this.logger.info('Loadout manager cleaned up');
  }

  // Statistics and monitoring
  getStatistics(): LoadoutManagerStatistics {
    const validLoadouts = this.getValidLoadouts();
    
    return {
      totalLoadouts: this.loadouts.size,
      validLoadouts: validLoadouts.length,
      invalidLoadouts: this.loadouts.size - validLoadouts.length,
      directories: this.config.loadoutDirectories.length,
      hotReloadEnabled: this.config.enableHotReload,
      cacheEnabled: this.config.cacheEnabled,
      averageAgentsPerLoadout: validLoadouts.length > 0 
        ? validLoadouts.reduce((sum, l) => sum + l.agents.length, 0) / validLoadouts.length 
        : 0,
      averageStagesPerLoadout: validLoadouts.length > 0
        ? validLoadouts.reduce((sum, l) => sum + l.stages.length, 0) / validLoadouts.length
        : 0,
    };
  }
}

export interface LoadoutComparison {
  loadouts: string[];
  commonFeatures: string[];
  differences: string[];
  recommendations: string[];
}

export interface LoadoutManagerStatistics {
  totalLoadouts: number;
  validLoadouts: number;
  invalidLoadouts: number;
  directories: number;
  hotReloadEnabled: boolean;
  cacheEnabled: boolean;
  averageAgentsPerLoadout: number;
  averageStagesPerLoadout: number;
}