# typed: true
# frozen_string_literal: true

require "formula"

# Helper class for traversing a formula's previous versions.
#
# @api private
class FormulaVersions
  include Context

  IGNORED_EXCEPTIONS = [
    ArgumentError, NameError, SyntaxError, TypeError,
    FormulaSpecificationError, FormulaValidationError,
    ErrorDuringExecution, LoadError, MethodDeprecatedError
  ].freeze

  MAX_VERSIONS_DEPTH = 2

  attr_reader :name, :path, :repository, :entry_name

  def initialize(formula)
    @name = formula.name
    @path = formula.path
    @repository = formula.tap.path
    @entry_name = @path.relative_path_from(repository).to_s
    @current_formula = formula
    @formula_at_revision = {}
  end

  def rev_list(branch)
    repository.cd do
      Utils.popen_read("git", "rev-list", "--abbrev-commit", "--remove-empty", branch, "--", entry_name) do |io|
        yield io.readline.chomp until io.eof?
      end
    end
  end

  def file_contents_at_revision(rev)
    repository.cd { Utils.popen_read("git", "cat-file", "blob", "#{rev}:#{entry_name}") }
  end

  def formula_at_revision(rev)
    Homebrew.raise_deprecation_exceptions = true

    yield @formula_at_revision[rev] ||= begin
      contents = file_contents_at_revision(rev)
      nostdout { Formulary.from_contents(name, path, contents) }
    end
  rescue *IGNORED_EXCEPTIONS => e
    # We rescue these so that we can skip bad versions and
    # continue walking the history
    odebug "#{e} in #{name} at revision #{rev}", e.backtrace
  rescue FormulaUnavailableError
    nil
  ensure
    Homebrew.raise_deprecation_exceptions = false
  end

  def bottle_version_map(branch)
    map = Hash.new { |h, k| h[k] = [] }

    versions_seen = 0
    rev_list(branch) do |rev|
      formula_at_revision(rev) do |f|
        bottle = f.bottle_specification
        map[f.pkg_version] << bottle.rebuild unless bottle.checksums.empty?
        versions_seen = (map.keys + [f.pkg_version]).uniq.length
      end
      return map if versions_seen > MAX_VERSIONS_DEPTH
    rescue MacOSVersionError => e
      odebug "#{e} in #{name} at revision #{rev}"
      break
    end
    map
  end
end
