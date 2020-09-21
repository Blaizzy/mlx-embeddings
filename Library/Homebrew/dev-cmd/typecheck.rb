# frozen_string_literal: true

require "cli/parser"

module Homebrew
  module_function

  SORBET_FILES_YAML = "sorbet/files.yaml"

  def typecheck_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `typecheck`

        Check for typechecking errors using Sorbet.
      EOS
      switch "-q", "--quiet",
             description: "Silence all non-critical errors."
      switch "--update",
             description: "Update RBI files and prune #{SORBET_FILES_YAML}"
      switch "--fail-if-not-changed",
             description: "Return a failing status code if all gems are up to date " \
                          "and gem definitions do not need a tapioca update"
      flag   "--dir=",
             description: "Typecheck all files in a specific directory."
      flag   "--file=",
             description: "Typecheck a single file."
      flag   "--ignore=",
             description: "Ignores input files that contain the given string " \
                          "in their paths (relative to the input path passed to Sorbet)."
      conflicts "--dir", "--file"
      max_named 0
    end
  end

  def typecheck
    args = typecheck_args.parse

    Homebrew.install_bundler_gems!

    HOMEBREW_LIBRARY_PATH.cd do
      if args.update?
        ohai "Checking for deleted filenames in #{SORBET_FILES_YAML}..."
        lines_to_keep = []
        sorbet_keywords = ["true:", "false:", "strict:", "strong:"]

        File.readlines(SORBET_FILES_YAML).map(&:chomp).each do |line|
          if sorbet_keywords.include?(line) || line.blank?
            lines_to_keep << line
          elsif line.end_with?(".rb")
            filepath = line.split(" ").last
            lines_to_keep << line if File.exist?(filepath)
          end

          File.write(SORBET_FILES_YAML, "#{lines_to_keep.join("\n")}\n")
        end

        ohai "Updating Tapioca RBI files..."
        system "bundle", "exec", "tapioca", "sync"
        system "bundle", "exec", "srb", "rbi", "hidden-definitions"
        system "bundle", "exec", "srb", "rbi", "todo"

        Homebrew.failed = system("git", "diff", "--stat", "--exit-code") if args.fail_if_not_changed?

        return
      end

      srb_exec = %w[bundle exec srb tc]
      srb_exec << "--quiet" if args.quiet?
      srb_exec += ["--ignore", args.ignore] if args.ignore.present?
      if args.file.present? || args.dir.present?
        cd("sorbet")
        srb_exec += ["--file", "../#{args.file}"] if args.file
        srb_exec += ["--dir", "../#{args.dir}"] if args.dir
      else
        srb_exec += ["--typed-override", SORBET_FILES_YAML]
      end
      Homebrew.failed = !system(*srb_exec)
    end
  end
end
