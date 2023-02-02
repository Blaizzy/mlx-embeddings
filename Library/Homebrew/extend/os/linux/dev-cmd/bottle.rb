# typed: false
# frozen_string_literal: true

module Homebrew
  extend T::Sig

  module_function

  def setup_tar_and_args!(args, mtime)
    # Without --only-json-tab bottles are never reproducible
    default_tar_args = ["tar", [].freeze].freeze
    return default_tar_args unless args.only_json_tab?

    # Ensure tar is set up for reproducibility.
    # https://reproducible-builds.org/docs/archives/
    gnutar_args = [
      "--format", "pax", "--owner", "0", "--group", "0", "--sort", "name", "--mtime=#{mtime}",
      # Set exthdr names to exclude PID (for GNU tar <1.33). Also don't store atime and ctime.
      "--pax-option", "globexthdr.name=/GlobalHead.%n,exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime"
    ].freeze

    ["tar", gnutar_args].freeze
  end

  def formula_ignores(f)
    ignores = []
    cellar_regex = Regexp.escape(HOMEBREW_CELLAR)
    prefix_regex = Regexp.escape(HOMEBREW_PREFIX)

    # Ignore matches to go keg, because all go binaries are statically linked.
    any_go_deps = f.deps.any? do |dep|
      dep.name =~ Version.formula_optionally_versioned_regex(:go)
    end
    if any_go_deps
      go_regex = Version.formula_optionally_versioned_regex(:go, full: false)
      ignores << %r{#{cellar_regex}/#{go_regex}/[\d.]+/libexec}
    end

    ignores << case f.name
    # On Linux, GCC installation can be moved so long as the whole directory tree is moved together:
    # https://gcc-help.gcc.gnu.narkive.com/GnwuCA7l/moving-gcc-from-the-installation-path-is-it-allowed.
    when Version.formula_optionally_versioned_regex(:gcc)
      Regexp.union(%r{#{cellar_regex}/gcc}, %r{#{prefix_regex}/opt/gcc})
    # binutils is relocatable for the same reason: https://github.com/Homebrew/brew/pull/11899#issuecomment-906804451.
    when Version.formula_optionally_versioned_regex(:binutils)
      %r{#{cellar_regex}/binutils}
    end

    ignores.compact
  end
end
