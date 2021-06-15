# typed: strict
# frozen_string_literal: true

# Helper functions for working with paths
module Utils
  extend T::Sig

  # Checks if a a child path is a descendant of a given parent path
  sig { params(parent_path: T.any(String, Pathname), child_path: T.any(String, Pathname)).returns(T::Boolean) }
  def self.path_is_parent_of?(parent_path, child_path)
    parent_component_array = Pathname(parent_path).each_filename.to_a
    child_component_array = Pathname(child_path).each_filename.to_a

    child_component_array.first(parent_component_array.length) == parent_component_array
  end

  # Gets a condensed short brew path for a given path, or the original path if it cannot be condensed
  sig { params(long_path: T.any(String, Pathname)).returns(String) }
  def self.shortened_brew_path(long_path)
    short_path = long_path.to_s
    long_path = Pathname(long_path)

    if long_path.exist?
      begin
        k = Keg.for(long_path)
        opt_record = k.opt_record
        formula_name = k.to_formula.name
      rescue FormulaUnavailableError, NotAKegError
        nil
      else
        short_path = short_path.sub(/\A#{Regexp.escape(opt_record.to_s)}/, "$(brew --prefix #{formula_name})")
      end
    end

    try_paths = {
      HOMEBREW_CACHE      => "--cache",
      HOMEBREW_CELLAR     => "--cellar",
      HOMEBREW_REPOSITORY => "--repository",
      HOMEBREW_PREFIX     => "--prefix",
    }

    try_paths.each do |try_path, flag|
      if path_is_parent_of?(try_path, long_path)
        short_path = short_path.sub(/\A#{Regexp.escape(try_path.to_s)}/, "$(brew #{flag})")
        break
      end
    end

    short_path
  end
end
