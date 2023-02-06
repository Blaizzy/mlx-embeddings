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

    # Ensure tar is set up for reproducibility.
    # https://reproducible-builds.org/docs/archives/
    gnutar_args = [
      "--format", "pax", "--owner", "0", "--group", "0", "--sort", "name", "--mtime=#{mtime}",
      # Set exthdr names to exclude PID (for GNU tar <1.33). Also don't store atime and ctime.
      "--pax-option", "globexthdr.name=/GlobalHead.%n,exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime"
    ].freeze

    ["#{gnu_tar.opt_bin}/gtar", gnutar_args].freeze
  end
end
