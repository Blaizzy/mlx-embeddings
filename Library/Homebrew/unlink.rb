# typed: false
# frozen_string_literal: true

module Homebrew
  # Provides helper methods for unlinking formulae and kegs with consistent output.
  module Unlink
    module_function

    def unlink_versioned_formulae(formula, verbose: false)
      formula.versioned_formulae
             .select(&:keg_only?)
             .select(&:linked?)
             .map(&:any_installed_keg)
             .compact
             .select(&:directory?)
             .each do |keg|
        unlink(keg, verbose: verbose)
      end
    end

    def unlink(keg, dry_run: false, verbose: false)
      options = { dry_run: dry_run, verbose: verbose }

      keg.lock do
        print "Unlinking #{keg}... "
        puts if verbose
        puts "#{keg.unlink(**options)} symlinks removed."
      end
    end
  end
end
