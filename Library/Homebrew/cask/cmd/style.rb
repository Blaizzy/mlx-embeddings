# frozen_string_literal: true

require "json"

module Cask
  class Cmd
    class Style < AbstractCommand
      def self.description
        "Checks style of the given <cask> using RuboCop."
      end

      def self.parser
        super do
          switch "--fix",
                 description: "Fix style violations automatically using RuboCop's auto-correct feature."
        end
      end

      def self.rubocop(*paths, auto_correct: false, debug: false, json: false)
        Homebrew.install_bundler_gems!

        cache_env = { "XDG_CACHE_HOME" => "#{HOMEBREW_CACHE}/style" }
        hide_warnings = debug ? [] : [ENV["HOMEBREW_RUBY_PATH"], "-S"]

        args = [
          "--force-exclusion",
          "--config", "#{HOMEBREW_LIBRARY}/.rubocop_cask.yml"
        ]

        if json
          args << "--format" << "json"
        else
          if auto_correct
            args << "--auto-correct"
          else
            args << "--debug" if debug
            args << "--parallel"
          end

          args << "--format" << "simple"
          args << "--color" if Tty.color?
        end

        executable, *args = [*hide_warnings, "rubocop", *args, "--", *paths]

        result = Dir.mktmpdir do |tmpdir|
          system_command executable, args: args, chdir: tmpdir, env: cache_env,
                                     print_stdout: !json, print_stderr: !json
        end

        result.assert_success! unless (0..1).cover?(result.exit_status)

        return JSON.parse(result.stdout) if json

        result
      end

      def run
        result = self.class.rubocop(*cask_paths, auto_correct: args.fix?, debug: args.debug?)
        raise CaskError, "Style check failed." unless result.status.success?
      end

      def cask_paths
        @cask_paths ||= if args.named.empty?
          Tap.map(&:cask_dir).select(&:directory?).concat(test_cask_paths)
        elsif args.named.any? { |file| File.exist?(file) }
          args.named.map { |path| Pathname(path).expand_path }
        else
          casks.map(&:sourcefile_path)
        end
      end

      def test_cask_paths
        [
          Pathname.new("#{HOMEBREW_LIBRARY}/Homebrew/test/support/fixtures/cask/Casks"),
          Pathname.new("#{HOMEBREW_LIBRARY}/Homebrew/test/support/fixtures/third-party/Casks"),
        ]
      end
    end
  end
end
