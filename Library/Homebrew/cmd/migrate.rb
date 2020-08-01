# frozen_string_literal: true

require "migrator"
require "cli/parser"

module Homebrew
  module_function

  def migrate_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `migrate` [<options>] <formula>

        Migrate renamed packages to new names, where <formula> are old names of
        packages.
      EOS
      switch "-f", "--force",
             description: "Treat installed <formula> and provided <formula> as if they are from "\
                          "the same taps and migrate them anyway."

      min_named :formula
    end
  end

  def migrate
    args = migrate_args.parse

    args.resolved_formulae.each do |f|
      if f.oldname
        unless (rack = HOMEBREW_CELLAR/f.oldname).exist? && !rack.subdirs.empty?
          raise NoSuchKegError, f.oldname
        end
        raise "#{rack} is a symlink" if rack.symlink?
      end

      migrator = Migrator.new(f, force: args.force?)
      migrator.migrate
    end
  end
end
