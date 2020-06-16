# frozen_string_literal: true

require "formula"
require "cask/cask_loader"

module Readall
  class << self
    def valid_ruby_syntax?(ruby_files)
      failed = false
      ruby_files.each do |ruby_file|
        # As a side effect, print syntax errors/warnings to `$stderr`.
        failed = true if syntax_errors_or_warnings?(ruby_file)
      end
      !failed
    end

    def valid_aliases?(alias_dir, formula_dir)
      return true unless alias_dir.directory?

      failed = false
      alias_dir.each_child do |f|
        if !f.symlink?
          onoe "Non-symlink alias: #{f}"
          failed = true
        elsif !f.file?
          onoe "Non-file alias: #{f}"
          failed = true
        end

        if (formula_dir/"#{f.basename}.rb").exist?
          onoe "Formula duplicating alias: #{f}"
          failed = true
        end
      end
      !failed
    end

    def valid_formulae?(formulae)
      success = true
      formulae.each do |file|
        Formulary.factory(file)
      rescue Interrupt
        raise
      rescue Exception => e # rubocop:disable Lint/RescueException
        onoe "Invalid formula: #{file}"
        $stderr.puts e
        success = false
      end
      success
    end

    def valid_casks?(casks)
      success = true
      casks.each do |file|
        Cask::CaskLoader.load(file)
      rescue Interrupt
        raise
      rescue Exception => e # rubocop:disable Lint/RescueException
        onoe "Invalid cask: #{file}"
        $stderr.puts e
        success = false
      end
      success
    end

    def valid_tap?(tap, options = {})
      success = true
      if options[:aliases]
        valid_aliases = valid_aliases?(tap.alias_dir, tap.formula_dir)
        success = false unless valid_aliases
      end
      valid_formulae = valid_formulae?(tap.formula_files)
      valid_casks = valid_casks?(tap.cask_files)
      success = false if !valid_formulae || !valid_casks
      success
    end

    private

    def syntax_errors_or_warnings?(rb)
      # Retrieve messages about syntax errors/warnings printed to `$stderr`.
      _, err, status = system_command(RUBY_PATH, args: ["-c", "-w", rb], print_stderr: false)

      # Ignore unnecessary warning about named capture conflicts.
      # See https://bugs.ruby-lang.org/issues/12359.
      messages = err.lines
                    .grep_v(/named capture conflicts a local variable/)
                    .join

      $stderr.print messages

      # Only syntax errors result in a non-zero status code. To detect syntax
      # warnings we also need to inspect the output to `$stderr`.
      !status.success? || !messages.chomp.empty?
    end
  end
end

require "extend/os/readall"
