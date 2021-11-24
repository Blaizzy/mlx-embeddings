# typed: true
# frozen_string_literal: true

require "migrator"
require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def migrate_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Migrate renamed packages to new names, where <formula> are old names of
        packages.
      EOS
      switch "-f", "--force",
             description: "Treat installed <formula> and provided <formula> as if they are from "\
                          "the same taps and migrate them anyway."
      switch "-n", "--dry-run",
             description: "Show what would be migrated, but do not actually migrate anything."

      named_args :installed_formula, min: 1
    end
  end

  def migrate
    args = migrate_args.parse

    args.named.to_kegs.each do |keg|
      f = Formulary.from_keg(keg)

      if f.oldname
        rack = HOMEBREW_CELLAR/f.oldname
        raise NoSuchKegError, f.oldname if !rack.exist? || rack.subdirs.empty?

        odie "#{rack} is a symlink" if rack.symlink?
      end

      Migrator.migrate_if_needed(f, force: args.force?, dry_run: args.dry_run?)
    end
  end
end
