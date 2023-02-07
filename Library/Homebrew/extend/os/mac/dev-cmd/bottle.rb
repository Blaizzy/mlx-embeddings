# typed: false
# frozen_string_literal: true

module Homebrew
  extend T::Sig

  module_function

  def setup_tar_and_args!(args, mtime)
    generic_setup_tar_and_args!(args, mtime)

    # Use gnu-tar on macOS as it can be set up for reproducibility better than libarchive.
    begin
      gnu_tar = Formula["gnu-tar"]
    rescue FormulaUnavailableError
      return default_tar_args
    end

    ensure_formula_installed!(gnu_tar, reason: "bottling")

    ["#{gnu_tar.opt_bin}/gtar", gnutar_args(mtime)].freeze
  end
end
