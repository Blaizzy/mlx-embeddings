class Formula
  module Compat
    # Run `scons` using a Homebrew-installed version rather than whatever is
    # in the `PATH`.
    def scons(*)
      odisabled("scons", 'system "scons"')
    end

    # Run `make` 3.81 or newer.
    # Uses the system make on Leopard and newer, and the
    # path to the actually-installed make on Tiger or older.
    def make(*)
      odisabled("make", 'system "make"')
    end
  end

  prepend Compat
end
