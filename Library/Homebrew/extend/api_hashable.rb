# typed: true
# frozen_string_literal: true

# Used to substitute common paths with generic placeholders when generating JSON for the API.
module APIHashable
  def generating_hash!
    return if generating_hash?

    # Apply monkeypatches for API generation
    @old_homebrew_prefix = HOMEBREW_PREFIX
    @old_homebrew_cellar = HOMEBREW_CELLAR
    @old_home = Dir.home
    Object.send(:remove_const, :HOMEBREW_PREFIX)
    Object.const_set(:HOMEBREW_PREFIX, Pathname.new(HOMEBREW_PREFIX_PLACEHOLDER))
    Object.const_set(:HOMEBREW_CELLAR, Pathname.new(HOMEBREW_CELLAR_PLACEHOLDER))
    ENV["HOME"] = HOMEBREW_HOME_PLACEHOLDER

    @generating_hash = true
  end

  def generated_hash!
    return unless generating_hash?

    # Revert monkeypatches for API generation
    Object.send(:remove_const, :HOMEBREW_PREFIX)
    Object.const_set(:HOMEBREW_PREFIX, @old_homebrew_prefix)
    Object.send(:remove_const, :HOMEBREW_CELLAR)
    Object.const_set(:HOMEBREW_CELLAR, @old_homebrew_cellar)
    ENV["HOME"] = @old_home

    @generating_hash = false
  end

  def generating_hash?
    @generating_hash ||= false
    @generating_hash == true
  end
end
