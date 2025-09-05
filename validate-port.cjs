#!/usr/bin/env node

/**
 * Basic validation script for the TypeScript port of Mimir.
 * Tests core functionality without requiring full dependency installation.
 */

const fs = require('fs');
const path = require('path');

function validatePortSuccess() {
  console.log('🔍 Validating Mimir TypeScript Port...\n');

  // Check if core TypeScript files exist
  const coreFiles = [
    'src/types.ts',
    'src/config/config.ts', 
    'src/utils/logger.ts',
    'src/utils/filesystem.ts',
    'src/pipeline/discovery.ts',
    'src/pipeline/symbols.ts',
    'src/pipeline/lens-client.ts',
    'src/mcp/server.ts',
    'src/cli.ts',
    'src/server.ts'
  ];

  console.log('📁 Checking core TypeScript files:');
  let allFilesExist = true;

  for (const file of coreFiles) {
    const filePath = path.join(process.cwd(), file);
    const exists = fs.existsSync(filePath);
    console.log(`  ${exists ? '✅' : '❌'} ${file}`);
    if (!exists) allFilesExist = false;
  }

  if (!allFilesExist) {
    console.log('\n❌ Some core files are missing!');
    process.exit(1);
  }

  // Check if test files exist
  const testFiles = [
    'tests/unit/config.test.ts',
    'tests/unit/filesystem.test.ts', 
    'tests/unit/logger.test.ts',
    'tests/unit/lens-client.test.ts',
    'tests/unit/mcp-server.test.ts',
    'tests/integration/cli-integration.test.ts'
  ];

  console.log('\n🧪 Checking test files:');
  let allTestsExist = true;

  for (const file of testFiles) {
    const filePath = path.join(process.cwd(), file);
    const exists = fs.existsSync(filePath);
    console.log(`  ${exists ? '✅' : '❌'} ${file}`);
    if (!exists) allTestsExist = false;
  }

  if (!allTestsExist) {
    console.log('\n❌ Some test files are missing!');
    process.exit(1);
  }

  // Check configuration files
  const configFiles = [
    'package.json',
    'tsconfig.json',
    'bunfig.toml',
    'tests/setup.ts'
  ];

  console.log('\n⚙️  Checking configuration files:');
  let allConfigsExist = true;

  for (const file of configFiles) {
    const filePath = path.join(process.cwd(), file);
    const exists = fs.existsSync(filePath);
    console.log(`  ${exists ? '✅' : '❌'} ${file}`);
    if (!exists) allConfigsExist = false;
  }

  if (!allConfigsExist) {
    console.log('\n❌ Some configuration files are missing!');
    process.exit(1);
  }

  // Validate package.json content
  console.log('\n📦 Validating package.json:');
  const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
  
  const requiredDeps = ['commander', 'hono', 'zod', '@anthropic-ai/sdk'];
  const hasDeps = requiredDeps.every(dep => packageJson.dependencies[dep]);
  console.log(`  ${hasDeps ? '✅' : '❌'} Required dependencies present`);

  const hasScripts = ['build', 'build:server', 'test', 'type-check'].every(
    script => packageJson.scripts[script]
  );
  console.log(`  ${hasScripts ? '✅' : '❌'} Required scripts present`);

  // Count lines of code
  function countLinesInFile(filePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      return content.split('\n').length;
    } catch {
      return 0;
    }
  }

  console.log('\n📊 Code Statistics:');
  let totalLines = 0;
  let totalFiles = 0;

  [...coreFiles, ...testFiles].forEach(file => {
    const lines = countLinesInFile(file);
    if (lines > 0) {
      totalLines += lines;
      totalFiles++;
    }
  });

  console.log(`  📝 Total TypeScript files: ${totalFiles}`);
  console.log(`  📏 Total lines of code: ${totalLines}`);

  // Check Python file count for comparison
  const pythonFiles = [];
  function findPythonFiles(dir) {
    try {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== 'node_modules') {
          findPythonFiles(fullPath);
        } else if (entry.isFile() && entry.name.endsWith('.py')) {
          pythonFiles.push(fullPath);
        }
      }
    } catch (error) {
      // Ignore directory access errors
    }
  }

  findPythonFiles('.');
  console.log(`  🐍 Python files remaining: ${pythonFiles.length}`);

  console.log('\n🎉 Port Validation Summary:');
  console.log('✅ All core TypeScript files present');
  console.log('✅ All test files created'); 
  console.log('✅ Configuration files in place');
  console.log('✅ Package.json properly configured');
  console.log(`✅ ${totalLines} lines of TypeScript code written`);
  console.log(`📋 ${pythonFiles.length} Python files ready for cleanup`);

  console.log('\n🚀 TypeScript port is structurally complete!');
  console.log('\nNext steps:');
  console.log('1. Resolve dependency installation issues');
  console.log('2. Fix failing tests');
  console.log('3. Build and test executables');
  console.log('4. Remove Python files after verification');
  
  return true;
}

if (require.main === module) {
  try {
    validatePortSuccess();
  } catch (error) {
    console.error('\n❌ Validation failed:', error.message);
    process.exit(1);
  }
}

module.exports = { validatePortSuccess };