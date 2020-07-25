# frozen_string_literal: true

module SharedEnvExtension
  # @private
  def effective_arch
    if @args&.build_bottle? && @args&.bottle_arch
      @args.bottle_arch.to_sym
    elsif @args&.build_bottle?
      Hardware.oldest_cpu
    else
      :native
    end
  end
end
